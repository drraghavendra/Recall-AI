(() => {
  const TARGET_RATE = 16000;
  const FRAME_PERIOD_MS = 1000;
  // Webcams start dark while auto-exposure settles; frames sent right away gave black saved photos.
  const CAMERA_WARMUP_MS = 2000;
  const AUDIO_CHUNK_SAMPLES = 1600;
  // Lip check: while the mic hears sound, send small frames at ~5 fps so the backend can tell whether
  // the person in focus is the one talking.
  const LIP_PERIOD_MS = 200;
  const LIP_HOLD_MS = 1500;
  const LIP_SOUND_RMS = 0.01;
  // People cards: at most this many; a newcomer replaces someone only once we know something about them.
  const MAX_CARDS = 3;
  const LISTENING_CLEAR_MS = 150_000;
  const $ = (id) => document.getElementById(id);

  const video = $("video");
  const canvas = document.createElement("canvas");
  const context2d = canvas.getContext("2d");
  const lipCanvas = document.createElement("canvas");
  const lipContext = lipCanvas.getContext("2d");
  let lipCheck = false;
  let lipTimer = null;
  let lipUntil = 0;
  // faceId -> { el, lastSeen, heard, listeningTimer, heardTimer, hasPhoto }, in on-screen slot order.
  let cards = new Map();
  // faceId -> persona, or null when we have none yet; missing = not fetched.
  const personas = new Map();
  const admitting = new Set();
  let present = new Set();
  let focusedId = null;
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
  let voiceParts = [];
  let voiceTimer = null;
  let transcriptCount = 0;
  let toastTimer = null;

  const personName = (faceId) => {
    const persona = personas.get(faceId);
    if (persona && (persona.nickname || (persona.names || [])[0])) return persona.nickname || persona.names[0];
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

  const GROUPS = [["work", "Work"], ["family", "Family"], ["interests", "Likes"], ["plans", "Coming up"], ["other", "Also"]];
  const factsOf = (persona) => (persona ? GROUPS.flatMap(([key]) => persona[key] || []) : []);
  const goodPersona = (persona) => Boolean(persona && ((persona.names || []).length || factsOf(persona).length));
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const personaRequests = new Map();

  async function loadPersona(faceId) {
    if (personas.has(faceId)) return personas.get(faceId);
    if (!personaRequests.has(faceId)) {
      personaRequests.set(faceId, (async () => {
        try {
          const response = await fetch(`/persona/${encodeURIComponent(faceId)}`);
          const persona = response.ok ? await response.json() : null;
          // A persona_updated push may have arrived while this was in flight; it's newer, so keep it.
          if (!personas.has(faceId)) personas.set(faceId, persona);
        } catch {
          // Leave it unset so the next face update tries again.
        } finally {
          personaRequests.delete(faceId);
        }
      })());
    }
    await personaRequests.get(faceId);
    return personas.get(faceId) ?? null;
  }

  function seenText(card, faceId) {
    if (present.has(faceId)) return "Here now";
    const minutes = Math.round((performance.now() - card.lastSeen) / 60000);
    return minutes < 1 ? "Seen just now" : minutes === 1 ? "Seen a minute ago" : `Seen ${minutes} minutes ago`;
  }

  function renderCardBody(faceId, fresh = []) {
    const card = cards.get(faceId);
    if (!card) return;
    const persona = personas.get(faceId);
    card.el.querySelector(".card-name").textContent = personName(faceId);
    const relation = persona && persona.relation && persona.relation !== "unknown" ? persona.relation : "";
    card.el.querySelector(".card-relation").textContent = relation;
    const body = card.el.querySelector(".card-body");
    body.replaceChildren();
    if (persona === undefined) {
      body.append(el("p", "card-muted", "Remembering…"));
      return;
    }
    if (!goodPersona(persona)) {
      body.append(el("p", "first-meeting", "Meeting for the first time."));
      return;
    }
    if (persona.summary) body.append(el("p", "card-summary", persona.summary));
    if (persona.last_talked_about) {
      const last = el("p", "card-last");
      last.append(el("strong", "", "Last time: "), document.createTextNode(persona.last_talked_about));
      body.append(last);
    }
    const groups = el("div", "card-groups");
    for (const [key, label] of GROUPS) {
      const facts = persona[key] || [];
      if (!facts.length) continue;
      const group = el("div", "card-group");
      group.append(el("div", "card-group-label", label));
      const list = el("ul");
      for (const fact of facts) list.append(el("li", fresh.includes(fact) ? "fresh" : "", fact));
      group.append(list);
      groups.append(group);
    }
    body.append(groups);
  }

  function setCardPhoto(faceId) {
    const card = cards.get(faceId);
    if (!card || card.hasPhoto) return;
    const image = card.el.querySelector("img");
    image.onload = () => { image.hidden = false; card.hasPhoto = true; card.el.querySelector(".forget-button").hidden = false; };
    image.onerror = () => { image.hidden = true; };
    image.src = `/faces/${encodeURIComponent(faceId)}/portrait?v=${Date.now()}`;
  }

  function buildCard(faceId) {
    const card = el("article", "person-card");
    const top = el("div", "card-top");
    const avatar = el("div", "avatar");
    const image = el("img");
    image.alt = "";
    image.hidden = true;
    avatar.append(el("div", "avatar-placeholder"), image);
    const id = el("div", "card-id");
    const presence = el("div", "presence");
    presence.append(el("i"), el("span", "presence-text"));
    id.append(el("div", "card-name"), el("div", "card-relation"), presence);
    top.append(avatar, id);
    const listening = el("div", "listening");
    listening.hidden = true;
    const dots = el("span", "listening-dots");
    dots.append(el("i"), el("i"), el("i"));
    listening.append(dots, el("span", "listening-text"));
    const heard = el("div", "heard");
    heard.hidden = true;
    heard.append(el("div", "heard-title", "Just learned"), el("ul"));
    const forget = el("button", "forget-button", "Forget this photo");
    forget.hidden = true;
    forget.addEventListener("click", () => void forgetPortrait(faceId));
    card.append(top, listening, heard, el("div", "card-body"), forget);
    return card;
  }

  function addCard(faceId, replacing = null) {
    const entry = { el: buildCard(faceId), lastSeen: performance.now(), heard: 0, hasPhoto: false };
    if (replacing && cards.has(replacing)) {
      // Keep the slot position so the other cards don't move.
      cards.get(replacing).el.replaceWith(entry.el);
      cards = new Map([...cards].map(([id, card]) => (id === replacing ? [faceId, entry] : [id, card])));
    } else {
      $("people").append(entry.el);
      cards.set(faceId, entry);
    }
    renderCardBody(faceId);
    setCardPhoto(faceId);
    void loadPersona(faceId).then(() => renderCardBody(faceId));
    showToast(`${personName(faceId)} is here`);
  }

  async function admit(faceId) {
    if (cards.has(faceId) || admitting.has(faceId)) return;
    if (cards.size < MAX_CARDS) return addCard(faceId);
    admitting.add(faceId);
    try {
      const persona = await loadPersona(faceId);
      if (cards.has(faceId) || !present.has(faceId) || !goodPersona(persona)) return;
      const away = [...cards].filter(([id]) => !present.has(id)).sort((a, b) => a[1].lastSeen - b[1].lastSeen);
      if (away.length) addCard(faceId, away[0][0]);
    } finally {
      admitting.delete(faceId);
      updateCards();
    }
  }

  function updateCards() {
    const now = performance.now();
    for (const [faceId, card] of cards) {
      const here = present.has(faceId);
      if (here) card.lastSeen = now;
      card.el.classList.toggle("here", here);
      card.el.classList.toggle("away", !here);
      card.el.classList.toggle("focused", faceId === focusedId);
      card.el.querySelector(".presence-text").textContent = seenText(card, faceId);
    }
    $("people-empty").hidden = cards.size > 0;
    const here = [...cards.keys()].filter((id) => present.has(id)).length;
    $("people-count").textContent = cards.size ? (here ? `${here} here now` : "No one here now") : "No one yet";
    $("people-count").classList.toggle("recognized", here > 0);
    const waiting = [...present].filter((id) => !cards.has(id));
    updateSpecsHud();
    $("waiting").hidden = !waiting.length;
    $("waiting").textContent = waiting.length ? `Getting to know ${waiting.map(personName).join(", ")}…` : "";
  }

  // Glasses view: frames the camera like the wearer's spectacles, with a heads-up label in the right lens.
  function updateSpecsHud() {
    const persona = focusedId ? personas.get(focusedId) : null;
    $("specs-hud").classList.toggle("idle", !focusedId);
    $("specs-name").textContent = focusedId
      ? [personName(focusedId), persona && persona.relation && persona.relation !== "unknown" ? persona.relation : ""].filter(Boolean).join(" · ")
      : "Looking…";
    $("specs-line").textContent = !focusedId ? "" : !goodPersona(persona) ? "Meeting for the first time"
      : persona.last_talked_about ? `Last time: ${persona.last_talked_about}` : persona.summary || "";
  }

  function setSpecs(on) {
    $("specs").hidden = !on;
    $("specs-toggle").setAttribute("aria-pressed", String(on));
    $("specs-toggle").textContent = on ? "Glasses view: on" : "Glasses view";
    try { localStorage.setItem("recall-specs", on ? "1" : "0"); } catch {}
    updateSpecsHud();
  }

  function setListening(faceId, text, kind = "") {
    const card = cards.get(faceId);
    if (!card) return;
    clearTimeout(card.listeningTimer);
    const listening = card.el.querySelector(".listening");
    listening.hidden = !text;
    listening.className = `listening ${kind}`;
    listening.querySelector(".listening-text").textContent = text || "";
  }

  function showLearned(faceId, facts) {
    const card = cards.get(faceId);
    if (!card) return;
    const heard = card.el.querySelector(".heard");
    const list = heard.querySelector("ul");
    list.replaceChildren(...facts.map((fact) => el("li", "", fact)));
    heard.hidden = false;
    clearTimeout(card.heardTimer);
    card.heardTimer = setTimeout(() => { heard.hidden = true; }, 30000);
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
    const card = event.speaker === "other" && event.face_id ? cards.get(event.face_id) : null;
    if (card) {
      card.heard += 1;
      setListening(event.face_id, `Listening · heard ${card.heard} ${card.heard === 1 ? "thing" : "things"}`, "pulse");
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
      // Only the debounced fields drive the cards, so one blurry or covered frame changes nothing.
      const previousFocus = focusedId;
      present = new Set(event.present || []);
      focusedId = event.focused || null;
      for (const faceId of present) void admit(faceId);
      if (event.face_id && event.face_id === focusedId && event.portrait_url) setCardPhoto(focusedId);
      updateCards();
      if (focusedId !== previousFocus) appendLog(focusedId ? `${personName(focusedId)} is here` : "Looking for a familiar face");
    } else if (event.type === "providers_changed") {
      showProviders(event);
    } else if (event.type === "transcript") {
      addTranscript(event);
    } else if (event.type === "persona_updated") {
      const before = new Set(factsOf(personas.get(event.face_id)));
      personas.set(event.face_id, event.persona);
      const fresh = factsOf(event.persona).filter((fact) => !before.has(fact));
      if (cards.has(event.face_id)) {
        renderCardBody(event.face_id, fresh);
        if (fresh.length) {
          showLearned(event.face_id, fresh);
          showToast(`Got it: ${fresh[0]}`);
        }
        const card = cards.get(event.face_id);
        if (!card.el.querySelector(".listening").classList.contains("pulse")) setListening(event.face_id, "");
      } else if (present.has(event.face_id)) {
        void admit(event.face_id);
      }
      updateSpecsHud();
      appendLog(`Remembered something new about ${personName(event.face_id)}`);
    } else if (event.type === "conversation_started") {
      $("talking").innerHTML = "<i></i> Talking";
      $("talking").classList.add("active");
      const card = event.face_id ? cards.get(event.face_id) : null;
      if (card) {
        card.heard = 0;
        setListening(event.face_id, "Listening…");
      }
    } else if (event.type === "conversation_ended") {
      $("talking").innerHTML = "<i></i> Quiet";
      $("talking").classList.remove("active");
      const card = event.face_id ? cards.get(event.face_id) : null;
      if (card && card.heard) {
        setListening(event.face_id, "Remembering what you talked about…", "saving");
        card.listeningTimer = setTimeout(() => setListening(event.face_id, ""), LISTENING_CLEAR_MS);
      } else if (card) {
        setListening(event.face_id, "");
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
      frameTimer = setTimeout(() => {
        if (!media) return;
        sendFrame();
        frameTimer = setInterval(sendFrame, FRAME_PERIOD_MS);
      }, CAMERA_WARMUP_MS);
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
    if (frameTimer) { clearTimeout(frameTimer); clearInterval(frameTimer); }
    frameTimer = null;
    stopLipBurst();
    present = new Set();
    focusedId = null;
    updateCards();
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

  async function forgetPortrait(faceId) {
    const card = cards.get(faceId);
    if (!card || !card.hasPhoto) return;
    if (!window.confirm(`Forget the photo of ${personName(faceId)}?`)) return;
    try {
      const response = await fetch(`/faces/${encodeURIComponent(faceId)}/portrait`, { method: "DELETE" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      $("consent").checked = false;
      card.hasPhoto = false;
      const image = card.el.querySelector("img");
      image.hidden = true;
      image.removeAttribute("src");
      card.el.querySelector(".forget-button").hidden = true;
      appendLog(`Photo of ${personName(faceId)} forgotten`);
    } catch (error) {
      appendLog("Couldn’t forget that photo, please try again");
    }
  }

  $("consent").addEventListener("change", () => { $("start").disabled = !$("consent").checked || !!media; });
  $("start").addEventListener("click", beginCapture);
  $("stop").addEventListener("click", () => void stopCapture());
  $("voice").addEventListener("click", enrollVoice);
  setInterval(updateCards, 15000);
  $("specs-toggle").addEventListener("click", () => setSpecs($("specs").hidden));
  try { if (localStorage.getItem("recall-specs") === "1") setSpecs(true); } catch {}
  window.addEventListener("beforeunload", () => { if (media) void stopCapture(false); });

  for (const id of Object.keys(PROVIDER_SELECTS)) $(id).addEventListener("change", (event) => void switchProvider(event.target));
  fetch("/providers").then((response) => response.json()).then(showProviders).catch(() => {
    appendLog("Could not read the active face and speech modules.");
  });
  fetch("/health").then((response) => response.json()).then((health) => { lipCheck = Boolean(health.lip_check); }).catch(() => {});
  })();
