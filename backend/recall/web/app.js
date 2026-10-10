(() => {
  const TARGET_RATE = 16000;
  const FRAME_PERIOD_MS = 1000;
  const AUDIO_CHUNK_SAMPLES = 1600;
  // Lip check: while the mic hears sound, send small frames at ~5 fps so the backend can tell whether
  // the person in focus is the one talking.
  const LIP_PERIOD_MS = 200;
  const LIP_HOLD_MS = 1500;
  const LIP_SOUND_RMS = 0.01;
  const OTHER_FACE_TTL_MS = 4000;
  const $ = (id) => document.getElementById(id);

  const video = $("video");
  const canvas = document.createElement("canvas");
  const context2d = canvas.getContext("2d");
  const lipCanvas = document.createElement("canvas");
  const lipContext = lipCanvas.getContext("2d");
  let lipCheck = false;
  let lipTimer = null;
  let lipUntil = 0;
  const others = new Map();
  // "You're being heard": facts already on screen for the focused person, so updates can highlight what's new.
  let shownFacts = new Set();
  let heardCount = 0;
  let heardTimer = null;
  let listeningTimer = null;
  let media = null;
  let socket = null;
  let audioContext = null;
  let source = null;
  let processor = null;
  let mutedOutput = null;
  let frameTimer = null;
  let pendingAudio = new Float32Array(0);
  let audioEpoch = 0;
  let sentSamples = 0;
  let currentFaceId = null;
  let lastPortraitUrl = null;
  let portraitRequest = 0;
  let voiceParts = [];
  let voiceTimer = null;
  let transcriptCount = 0;
  let toastTimer = null;

  const personName = (faceId) => {
    const match = /^person_(\d+)$/i.exec(faceId || "");
    return match ? `Person ${Number(match[1])}` : faceId ? `Person ${faceId.slice(0, 8)}` : "No one yet";
  };

  function setStatus(text, kind = "") {
    const connection = $("connection");
    connection.className = `connection ${kind}`;
    connection.innerHTML = `<i></i> ${text}`;
  }

  function appendLog(text) {
    $("event-log").textContent = `${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · ${text}`;
  }

  function showToast(text) {
    const toast = $("recognition-toast");
    toast.textContent = text;
    toast.classList.remove("show");
    void toast.offsetWidth;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 2700);
  }

  function bytesToBase64(bytes) {
    let binary = "";
    for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return btoa(binary);
  }

  function pcm16(samples) {
    const bytes = new Uint8Array(samples.length * 2);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < samples.length; i += 1) {
      const value = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(i * 2, value < 0 ? value * 32768 : value * 32767, true);
    }
    return bytes;
  }

  function resample(input, inputRate) {
    if (inputRate === TARGET_RATE) return input.slice();
    const length = Math.floor((input.length * TARGET_RATE) / inputRate);
    const output = new Float32Array(length);
    const step = inputRate / TARGET_RATE;
    for (let i = 0; i < length; i += 1) {
      const position = i * step;
      const left = Math.floor(position);
      const fraction = position - left;
      const right = Math.min(left + 1, input.length - 1);
      output[i] = input[left] * (1 - fraction) + input[right] * fraction;
    }
    return output;
  }

  function sendAudio(pcm) {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const ts = audioEpoch + sentSamples / TARGET_RATE;
    socket.send(JSON.stringify({ type: "audio", ts, pcm: bytesToBase64(pcm) }));
    sentSamples += pcm.length / 2;
    if (voiceTimer && voiceParts) voiceParts.push(pcm.slice());
  }

  function processAudio(event) {
    const incoming = resample(event.inputBuffer.getChannelData(0), audioContext.sampleRate);
    const joined = new Float32Array(pendingAudio.length + incoming.length);
    joined.set(pendingAudio);
    joined.set(incoming, pendingAudio.length);
    let offset = 0;
    while (joined.length - offset >= AUDIO_CHUNK_SAMPLES) {
      sendAudio(pcm16(joined.subarray(offset, offset + AUDIO_CHUNK_SAMPLES)));
      offset += AUDIO_CHUNK_SAMPLES;
    }
    pendingAudio = joined.slice(offset);
    if (lipCheck && rms(incoming) > LIP_SOUND_RMS) startLipBurst();
    event.outputBuffer.getChannelData(0).fill(0);
  }

  function rms(samples) {
    let sum = 0;
    for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
    return Math.sqrt(sum / Math.max(1, samples.length));
  }

  function startLipBurst() {
    lipUntil = performance.now() + LIP_HOLD_MS;
    if (!lipTimer) lipTimer = setInterval(sendLipFrame, LIP_PERIOD_MS);
  }

  function stopLipBurst() {
    if (lipTimer) clearInterval(lipTimer);
    lipTimer = null;
  }

  function sendLipFrame() {
    if (performance.now() > lipUntil) return stopLipBurst();
    if (!socket || socket.readyState !== WebSocket.OPEN || video.readyState < 2 || !video.videoWidth) return;
    if (socket.bufferedAmount > 1_000_000) return;
    const scale = Math.min(1, 640 / video.videoWidth);
    lipCanvas.width = Math.round(video.videoWidth * scale);
    lipCanvas.height = Math.round(video.videoHeight * scale);
    lipContext.drawImage(video, 0, 0, lipCanvas.width, lipCanvas.height);
    const jpeg = lipCanvas.toDataURL("image/jpeg", 0.75).split(",")[1];
    socket.send(JSON.stringify({ type: "lip_frames", ts: Date.now() / 1000, frame: jpeg }));
  }

  function sendFrame() {
    if (!socket || socket.readyState !== WebSocket.OPEN || video.readyState < 2 || !video.videoWidth) return;
    if (socket.bufferedAmount > 4_000_000) return;
    const scale = Math.min(1, 800 / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    context2d.drawImage(video, 0, 0, canvas.width, canvas.height);
    const jpeg = canvas.toDataURL("image/jpeg", 0.82).split(",")[1];
    socket.send(JSON.stringify({ type: "frames", ts: Date.now() / 1000, frames: [jpeg], store_portrait: $("consent").checked }));
  }

  async function fetchPersona(faceId) {
    const request = ++portraitRequest;
    $("memory-state").textContent = "Remembering…";
    try {
      const response = await fetch(`/persona/${encodeURIComponent(faceId)}`);
      if (request !== portraitRequest || faceId !== currentFaceId) return;
      if (response.status === 404) {
        const root = $("persona");
        root.replaceChildren();
        const first = document.createElement("p");
        first.className = "first-meeting";
        first.textContent = "Meeting for the first time.";
        root.append(first);
        $("memory-state").textContent = "First meeting";
        return;
      }
      if (!response.ok) throw new Error(`Memory request returned ${response.status}`);
      const persona = await response.json();
      renderPersona(persona);
      shownFacts = new Set(persona.key_facts || []);
      $("memory-state").textContent = "";
    } catch (error) {
      if (request === portraitRequest) {
        $("persona").textContent = "Couldn’t remember right now.";
        $("memory-state").textContent = "Unavailable";
      }
    }
  }

  function renderPersona(persona, fresh = []) {
    const root = $("persona");
    root.replaceChildren();
    const facts = persona.key_facts || [];
    if (facts.length) {
      const list = document.createElement("ul");
      for (const fact of facts) {
        const item = document.createElement("li");
        item.textContent = fact;
        if (fresh.includes(fact)) item.className = "fresh";
        list.append(item);
      }
      root.append(list);
    }
    if (persona.summary) {
      const summary = document.createElement("p");
      summary.className = "summary";
      summary.textContent = persona.summary;
      root.append(summary);
    }
    if (!facts.length && !persona.summary) root.textContent = "No saved details yet.";
  }

  function setListening(text, kind = "") {
    clearTimeout(listeningTimer);
    const el = $("listening");
    el.hidden = !text;
    el.className = `listening ${kind}`;
    $("listening-text").textContent = text || "";
  }

  function showHeard(facts) {
    const list = $("heard-list");
    list.replaceChildren();
    for (const fact of facts) {
      const item = document.createElement("li");
      item.textContent = fact;
      list.append(item);
    }
    $("heard").hidden = false;
    clearTimeout(heardTimer);
    heardTimer = setTimeout(() => { $("heard").hidden = true; }, 25000);
  }

  function personaLine(persona) {
    if (!persona) return "Meeting for the first time";
    return persona.summary || (persona.key_facts || [])[0] || "No saved details yet.";
  }

  async function loadOtherPersona(faceId) {
    try {
      const response = await fetch(`/persona/${encodeURIComponent(faceId)}`);
      const persona = response.ok ? await response.json() : null;
      updateOther(faceId, persona);
    } catch {
      updateOther(faceId, undefined);
    }
  }

  function updateOther(faceId, persona) {
    const entry = others.get(faceId);
    if (!entry) return;
    entry.el.querySelector(".other-memory").textContent =
      persona === undefined ? "Could not load memories" : personaLine(persona);
  }

  function renderOthers(list) {
    const now = performance.now();
    for (const other of list || []) {
      if (!other.face_id || other.face_id === currentFaceId) continue;
      let entry = others.get(other.face_id);
      if (!entry) {
        const el = document.createElement("div");
        el.className = "other-person";
        const name = document.createElement("div");
        name.className = "other-name";
        name.textContent = personName(other.face_id);
        const memory = document.createElement("div");
        memory.className = "other-memory";
        memory.textContent = "Recalling…";
        el.append(name, memory);
        $("others-list").append(el);
        entry = { el };
        others.set(other.face_id, entry);
        void loadOtherPersona(other.face_id);
      }
      entry.lastSeen = now;
    }
    pruneOthers();
  }

  function pruneOthers() {
    const now = performance.now();
    for (const [faceId, entry] of others) {
      if (faceId === currentFaceId || now - entry.lastSeen > OTHER_FACE_TTL_MS) {
        entry.el.remove();
        others.delete(faceId);
      }
    }
    $("others").hidden = others.size === 0;
    $("others-count").textContent = `${others.size} ${others.size === 1 ? "person" : "people"}`;
  }

  function showPortrait(url) {
    if (!url || url === lastPortraitUrl) return;
    lastPortraitUrl = url;
    const image = $("portrait");
    image.onload = () => {
      image.hidden = false;
      $("portrait-placeholder").classList.add("hidden");
      $("portrait-state").textContent = "";
      $("forget-portrait").disabled = false;
    };
    image.onerror = () => {
      image.hidden = true;
      $("portrait-placeholder").classList.remove("hidden");
      $("portrait-state").textContent = "";
      $("forget-portrait").disabled = true;
      lastPortraitUrl = null;
    };
    image.src = `${url}?v=${Date.now()}`;
  }

  function clearIdentity() {
    currentFaceId = null;
    lastPortraitUrl = null;
    portraitRequest += 1;
    $("face-name").textContent = "No one yet";
    $("face-id").textContent = "";
    $("confidence").textContent = "";
    $("match-status").textContent = "Waiting";
    $("match-status").classList.remove("recognized");
    $("portrait").hidden = true;
    $("portrait").removeAttribute("src");
    $("portrait-placeholder").classList.remove("hidden");
    $("portrait-state").textContent = "";
    $("forget-portrait").disabled = true;
    $("persona").textContent = "No one is with you right now.";
    $("memory-state").textContent = "";
    shownFacts = new Set();
    setListening("");
    $("heard").hidden = true;
  }

  function addTranscript(event) {
    const root = $("transcript");
    root.querySelector(".transcript-empty")?.remove();
    const line = document.createElement("div");
    const background = event.speaker === "background";
    line.className = `transcript-line ${event.speaker === "self" ? "self" : background ? "background" : "other"}`;
    const speaker = document.createElement("span");
    speaker.className = `transcript-speaker ${event.speaker === "self" ? "self" : background ? "background" : ""}`;
    speaker.textContent = event.speaker === "self" ? "You" : background ? "Nearby" : personName(event.face_id);
    if (background) speaker.title = event.reason === "too_quiet" ? "Too quiet to be the person in focus" : "The person in focus wasn’t moving their lips";
    const text = document.createElement("span");
    text.className = "transcript-text";
    text.textContent = event.text;
    line.append(speaker, text);
    root.append(line);
    while (root.querySelectorAll(".transcript-line").length > 24) root.querySelector(".transcript-line")?.remove();
    root.scrollTop = root.scrollHeight;
    transcriptCount += 1;
    $("transcript-count").textContent = `${transcriptCount} ${transcriptCount === 1 ? "line" : "lines"}`;
    if (event.speaker === "other" && event.face_id && event.face_id === currentFaceId) {
      heardCount += 1;
      setListening(`Listening · heard ${heardCount} ${heardCount === 1 ? "thing" : "things"}`, "pulse");
    }
  }

  const PROVIDER_SELECTS = { "face-backend": "face_backend", "stt-backend": "stt_backend" };

  function showProviders(state) {
    for (const [id, field] of Object.entries(PROVIDER_SELECTS)) {
      if (!state[field]) continue;
      $(id).value = state[field];
      $(id).dataset.current = state[field];
      $(id).disabled = false;
    }
    if (state.stt_backend) {
      // Sarvam is a cloud service, so don't claim the words stay on this computer when it's in use.
      $("stt-provider").textContent = state.stt_backend === "sarvam" ? "Written down using Sarvam (online)" : "Written down on this computer";
    }
  }

  async function switchProvider(select) {
    const field = PROVIDER_SELECTS[select.id];
    const label = select.selectedOptions[0].textContent;
    for (const id of Object.keys(PROVIDER_SELECTS)) $(id).disabled = true;
    appendLog(`Switching to ${label}… the first switch loads its model and can take a while.`);
    try {
      const response = await fetch("/providers", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [field]: select.value }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`);
      showProviders(body);
      appendLog(field === "face_backend"
        ? `Face module: ${label}. Faces saved by other modules are not recognized by this one.`
        : `Speech module: ${label}.`);
    } catch (error) {
      select.value = select.dataset.current;
      appendLog(error.message);
    } finally {
      for (const id of Object.keys(PROVIDER_SELECTS)) $(id).disabled = !$(id).dataset.current;
    }
  }

  function handleEvent(event) {
    if (event.type === "face_update") {
      if (event.face_id !== currentFaceId) {
        currentFaceId = event.face_id;
        if (currentFaceId) {
          const name = personName(currentFaceId);
          $("face-name").textContent = name;
          $("face-id").textContent = "";
          $("match-status").textContent = event.is_new ? "New friend" : "Recognized";
          $("match-status").classList.add("recognized");
          $("confidence").textContent = "";
          $("persona").textContent = "Remembering…";
          shownFacts = new Set();
          $("heard").hidden = true;
          $("memory-state").textContent = "Remembering…";
          $("portrait").hidden = true;
          $("portrait-placeholder").classList.remove("hidden");
          $("portrait-state").textContent = "";
          $("forget-portrait").disabled = true;
          void fetchPersona(currentFaceId);
          showToast(event.is_new ? `Meeting ${name}` : `This is ${name}`);
        } else {
          clearIdentity();
        }
      } else if (event.face_id) {
        $("confidence").textContent = "";
      }
      if (event.face_id) showPortrait(event.portrait_url);
      renderOthers(event.others);
      appendLog(event.face_id ? `${personName(event.face_id)} is here` : "Looking for a familiar face");
    } else if (event.type === "providers_changed") {
      showProviders(event);
    } else if (event.type === "transcript") {
      addTranscript(event);
    } else if (event.type === "persona_updated") {
      if (others.has(event.face_id)) updateOther(event.face_id, event.persona);
      if (event.face_id === currentFaceId) {
        const facts = event.persona.key_facts || [];
        const fresh = facts.filter((fact) => !shownFacts.has(fact));
        renderPersona(event.persona, fresh);
        shownFacts = new Set(facts);
        $("memory-state").textContent = "Just updated";
        if (fresh.length) {
          showHeard(fresh);
          showToast(`Got it: ${fresh[0]}`);
        }
        if (!$("listening").classList.contains("pulse")) setListening("");
      }
      appendLog("Remembered something new");
    } else if (event.type === "conversation_started") {
      $("talking").innerHTML = "<i></i> Talking";
      $("talking").classList.add("active");
      heardCount = 0;
      if (event.face_id && event.face_id === currentFaceId) setListening("Listening…");
    } else if (event.type === "conversation_ended") {
      $("talking").innerHTML = "<i></i> Quiet";
      $("talking").classList.remove("active");
      if (event.face_id && event.face_id === currentFaceId && heardCount) {
        setListening("Remembering what you talked about…", "saving");
        listeningTimer = setTimeout(() => setListening(""), 30000);
      } else {
        setListening("");
      }
    }
  }

  async function beginCapture() {
    if (!$("consent").checked) return;
    $("start").disabled = true;
    setStatus("Starting…", "");
    try {
      media = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        // Auto gain would make distant speakers as loud as the person in front; the backend uses loudness
        // to tell them apart.
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false },
      });
      video.srcObject = media;
      await video.play();
      $("placeholder").classList.add("hidden");

      socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
      socket.onmessage = (message) => {
        try { handleEvent(JSON.parse(message.data)); } catch (error) { appendLog("Something went wrong; still listening"); }
      };
      socket.onclose = () => setStatus("Stopped", "error");
      socket.onerror = () => setStatus("Can’t connect", "error");
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error("Backend connection timed out.")), 8000);
        socket.addEventListener("open", () => { clearTimeout(timeout); resolve(); }, { once: true });
        socket.addEventListener("error", () => { clearTimeout(timeout); reject(new Error("Could not connect to backend.")); }, { once: true });
      });

      audioContext = new AudioContext({ sampleRate: TARGET_RATE });
      await audioContext.resume();
      source = audioContext.createMediaStreamSource(media);
      processor = audioContext.createScriptProcessor(4096, 1, 1);
      mutedOutput = audioContext.createGain();
      mutedOutput.gain.value = 0;
      processor.onaudioprocess = processAudio;
      source.connect(processor);
      processor.connect(mutedOutput);
      mutedOutput.connect(audioContext.destination);
      audioEpoch = Date.now() / 1000;
      sentSamples = 0;
      pendingAudio = new Float32Array(0);
      sendFrame();
      frameTimer = setInterval(sendFrame, FRAME_PERIOD_MS);
      $("stop").disabled = false;
      $("voice").disabled = false;
      setStatus("On", "live");
      appendLog("Camera is on");
    } catch (error) {
      setStatus(error.message || "Could not start capture", "error");
      await stopCapture(false);
      $("start").disabled = !$("consent").checked;
    }
  }

  async function stopCapture(updateStatus = true) {
    if (frameTimer) clearInterval(frameTimer);
    frameTimer = null;
    stopLipBurst();
    for (const entry of others.values()) entry.el.remove();
    others.clear();
    pruneOthers();
    if (voiceTimer) {
      clearTimeout(voiceTimer);
      voiceTimer = null;
      voiceParts = null;
      $("voice").textContent = "Learn my voice";
    }
    if (processor) processor.disconnect();
    if (source) source.disconnect();
    if (mutedOutput) mutedOutput.disconnect();
    processor = source = mutedOutput = null;
    if (audioContext) await audioContext.close().catch(() => {});
    audioContext = null;
    if (media) media.getTracks().forEach((track) => track.stop());
    media = null;
    video.srcObject = null;
    $("placeholder").classList.remove("hidden");
    if (socket && socket.readyState < WebSocket.CLOSING) socket.close();
    socket = null;
    $("stop").disabled = true;
    $("voice").disabled = true;
    $("start").disabled = !$("consent").checked;
    if (updateStatus) {
      setStatus("Not started");
      appendLog("Camera is off");
    }
  }

  async function enrollVoice() {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    voiceParts = [];
    $("voice").disabled = true;
    $("voice").textContent = "Listening… keep talking";
    appendLog("Talk normally for about 10 seconds");
    voiceTimer = setTimeout(async () => {
      voiceTimer = null;
      const total = voiceParts.reduce((sum, part) => sum + part.length, 0);
      const audio = new Uint8Array(total);
      let offset = 0;
      for (const part of voiceParts) { audio.set(part, offset); offset += part.length; }
      voiceParts = null;
      $("voice").textContent = "Learn my voice";
      try {
        const response = await fetch("/self-voice", { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: audio });
        if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
        appendLog("Got it, I know your voice now");
      } catch (error) {
        appendLog("Couldn’t learn your voice, please try again");
      } finally {
        $("voice").disabled = !socket || socket.readyState !== WebSocket.OPEN;
      }
    }, 10_000);
  }

  async function forgetPortrait() {
    if (!currentFaceId || !lastPortraitUrl) return;
    if (!window.confirm(`Forget the photo of ${personName(currentFaceId)}?`)) return;
    try {
      const response = await fetch(`/faces/${encodeURIComponent(currentFaceId)}/portrait`, { method: "DELETE" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      $("consent").checked = false;
      lastPortraitUrl = null;
      $("portrait").hidden = true;
      $("portrait").removeAttribute("src");
      $("portrait-placeholder").classList.remove("hidden");
      $("portrait-state").textContent = "";
      $("forget-portrait").disabled = true;
      appendLog(`Photo of ${personName(currentFaceId)} forgotten`);
    } catch (error) {
      appendLog("Couldn’t forget that photo, please try again");
    }
  }

  $("consent").addEventListener("change", () => { $("start").disabled = !$("consent").checked || !!media; });
  $("start").addEventListener("click", beginCapture);
  $("stop").addEventListener("click", () => void stopCapture());
  $("voice").addEventListener("click", enrollVoice);
  $("forget-portrait").addEventListener("click", forgetPortrait);
  setInterval(pruneOthers, 1000);
  window.addEventListener("beforeunload", () => { if (media) void stopCapture(false); });

  for (const id of Object.keys(PROVIDER_SELECTS)) $(id).addEventListener("change", (event) => void switchProvider(event.target));
  fetch("/providers").then((response) => response.json()).then(showProviders).catch(() => {
    appendLog("Could not read the active face and speech modules.");
  });
  fetch("/health").then((response) => response.json()).then((health) => { lipCheck = Boolean(health.lip_check); }).catch(() => {});
  })();
