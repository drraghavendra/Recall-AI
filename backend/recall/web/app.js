(() => {
  const TARGET_RATE = 16000;
  const FRAME_PERIOD_MS = 5000;
  const AUDIO_CHUNK_SAMPLES = 1600;
  const $ = (id) => document.getElementById(id);

  const video = $("video");
  const canvas = document.createElement("canvas");
  const context2d = canvas.getContext("2d");
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
    return match ? `Person ${Number(match[1])}` : faceId ? `Person ${faceId.slice(0, 8)}` : "No one in focus";
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
    event.outputBuffer.getChannelData(0).fill(0);
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
    $("memory-state").textContent = "RECALLING";
    try {
      const response = await fetch(`/persona/${encodeURIComponent(faceId)}`);
      if (request !== portraitRequest || faceId !== currentFaceId) return;
      if (response.status === 404) {
        const root = $("persona");
        root.replaceChildren();
        const first = document.createElement("p");
        first.className = "first-meeting";
        first.textContent = "A new face. A new memory starts here.";
        root.append(first);
        $("memory-state").textContent = "NEW MEMORY";
        return;
      }
      if (!response.ok) throw new Error(`Memory request returned ${response.status}`);
      renderPersona(await response.json());
      $("memory-state").textContent = "ON FILE";
    } catch (error) {
      if (request === portraitRequest) {
        $("persona").textContent = `Could not load memories: ${error.message}`;
        $("memory-state").textContent = "OFFLINE";
      }
    }
  }

  function renderPersona(persona) {
    const root = $("persona");
    root.replaceChildren();
    const facts = persona.key_facts || [];
    if (facts.length) {
      const list = document.createElement("ul");
      for (const fact of facts) {
        const item = document.createElement("li");
        item.textContent = fact;
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

  function showPortrait(url) {
    if (!url || url === lastPortraitUrl) return;
    lastPortraitUrl = url;
    const image = $("portrait");
    image.onload = () => {
      image.hidden = false;
      $("portrait-placeholder").classList.add("hidden");
      $("portrait-state").textContent = "FRAME SAVED · LOCAL";
      $("forget-portrait").disabled = false;
    };
    image.onerror = () => {
      image.hidden = true;
      $("portrait-placeholder").classList.remove("hidden");
      $("portrait-state").textContent = "NO SAVED FRAME";
      $("forget-portrait").disabled = true;
      lastPortraitUrl = null;
    };
    image.src = `${url}?v=${Date.now()}`;
  }

  function clearIdentity() {
    currentFaceId = null;
    lastPortraitUrl = null;
    portraitRequest += 1;
    $("face-name").textContent = "No one in focus";
    $("face-id").textContent = "ID —";
    $("confidence").textContent = "—";
    $("match-status").textContent = "WAITING";
    $("match-status").classList.remove("recognized");
    $("portrait").hidden = true;
    $("portrait").removeAttribute("src");
    $("portrait-placeholder").classList.remove("hidden");
    $("portrait-state").textContent = "NO SAVED FRAME";
    $("forget-portrait").disabled = true;
    $("persona").textContent = "No one is in focus right now.";
    $("memory-state").textContent = "READY";
  }

  function addTranscript(event) {
    const root = $("transcript");
    root.querySelector(".transcript-empty")?.remove();
    const line = document.createElement("div");
    line.className = `transcript-line ${event.speaker === "self" ? "self" : "other"}`;
    const speaker = document.createElement("span");
    speaker.className = `transcript-speaker ${event.speaker === "self" ? "self" : ""}`;
    speaker.textContent = event.speaker === "self" ? "YOU" : personName(event.face_id);
    const text = document.createElement("span");
    text.className = "transcript-text";
    text.textContent = event.text;
    line.append(speaker, text);
    root.append(line);
    while (root.querySelectorAll(".transcript-line").length > 24) root.querySelector(".transcript-line")?.remove();
    root.scrollTop = root.scrollHeight;
    transcriptCount += 1;
    $("transcript-count").textContent = `${transcriptCount} ${transcriptCount === 1 ? "LINE" : "LINES"}`;
    showToast(event.speaker === "self" ? `You said · “${event.text}”` : `${personName(event.face_id)} · “${event.text}”`);
  }

  function handleEvent(event) {
    if (event.type === "face_update") {
      if (event.face_id !== currentFaceId) {
        currentFaceId = event.face_id;
        if (currentFaceId) {
          const name = personName(currentFaceId);
          $("face-name").textContent = name;
          $("face-id").textContent = `ID ${currentFaceId}`;
          $("match-status").textContent = event.is_new ? "NEW PERSON" : "RECOGNIZED";
          $("match-status").classList.add("recognized");
          $("confidence").textContent = `${Math.round(event.confidence * 100)}% match`;
          $("persona").textContent = "Looking for your saved memories…";
          $("memory-state").textContent = "RECALLING";
          $("portrait").hidden = true;
          $("portrait-placeholder").classList.remove("hidden");
          $("portrait-state").textContent = "CHECKING LOCAL MEMORY";
          $("forget-portrait").disabled = true;
          void fetchPersona(currentFaceId);
          showToast(event.is_new ? `${name} · new memory started` : `${name} · recognized`);
        } else {
          clearIdentity();
        }
      } else if (event.face_id) {
        $("confidence").textContent = `${Math.round(event.confidence * 100)}% match`;
      }
      if (event.face_id) showPortrait(event.portrait_url);
      appendLog(event.face_id ? `${personName(event.face_id)} recognized` : "No face currently in focus");
    } else if (event.type === "transcript") {
      addTranscript(event);
    } else if (event.type === "persona_updated") {
      if (event.face_id === currentFaceId) {
        renderPersona(event.persona);
        $("memory-state").textContent = "JUST UPDATED";
      }
      appendLog("Memory snapshot updated");
    } else if (event.type === "conversation_started") {
      $("talking").innerHTML = "<i></i> SPEAKING";
      $("talking").classList.add("active");
    } else if (event.type === "conversation_ended") {
      $("talking").innerHTML = "<i></i> IDLE";
      $("talking").classList.remove("active");
    }
  }

  async function beginCapture() {
    if (!$("consent").checked) return;
    $("start").disabled = true;
    setStatus("CONNECTING", "");
    try {
      const isPhone = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
      media = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: isPhone ? 1920 : 1280 },
          height: { ideal: isPhone ? 1080 : 720 },
          facingMode: { ideal: isPhone ? "environment" : "user" },
        },
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      video.srcObject = media;
      await video.play();
      $("placeholder").classList.add("hidden");

      socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
      socket.onmessage = (message) => {
        try { handleEvent(JSON.parse(message.data)); } catch (error) { appendLog(`Could not read backend event: ${error.message}`); }
      };
      socket.onclose = () => setStatus("DISCONNECTED", "error");
      socket.onerror = () => setStatus("CONNECTION ERROR", "error");
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
      setStatus("CONNECTED", "live");
      appendLog("Live view connected · portrait saving enabled");
    } catch (error) {
      setStatus(error.message || "Could not start capture", "error");
      await stopCapture(false);
      $("start").disabled = !$("consent").checked;
    }
  }

  async function stopCapture(updateStatus = true) {
    if (frameTimer) clearInterval(frameTimer);
    frameTimer = null;
    if (voiceTimer) {
      clearTimeout(voiceTimer);
      voiceTimer = null;
      voiceParts = null;
      $("voice").textContent = "Enroll my voice · 10 sec";
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
      setStatus("STANDBY");
      appendLog("Capture stopped · saved portrait remains on backend");
    }
  }

  async function enrollVoice() {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    voiceParts = [];
    $("voice").disabled = true;
    $("voice").textContent = "Recording… 10 sec";
    appendLog("Speak naturally for 10 seconds to enroll your own voice");
    voiceTimer = setTimeout(async () => {
      voiceTimer = null;
      const total = voiceParts.reduce((sum, part) => sum + part.length, 0);
      const audio = new Uint8Array(total);
      let offset = 0;
      for (const part of voiceParts) { audio.set(part, offset); offset += part.length; }
      voiceParts = null;
      $("voice").textContent = "Enroll my voice · 10 sec";
      try {
        const response = await fetch("/self-voice", { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: audio });
        if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
        appendLog("Voice profile saved");
      } catch (error) {
        appendLog(`Voice enrollment failed: ${error.message}`);
      } finally {
        $("voice").disabled = !socket || socket.readyState !== WebSocket.OPEN;
      }
    }, 10_000);
  }

  async function forgetPortrait() {
    if (!currentFaceId || !lastPortraitUrl) return;
    if (!window.confirm(`Forget the saved portrait for ${personName(currentFaceId)}?`)) return;
    try {
      const response = await fetch(`/faces/${encodeURIComponent(currentFaceId)}/portrait`, { method: "DELETE" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      $("consent").checked = false;
      lastPortraitUrl = null;
      $("portrait").hidden = true;
      $("portrait").removeAttribute("src");
      $("portrait-placeholder").classList.remove("hidden");
      $("portrait-state").textContent = "FRAME FORGOTTEN";
      $("forget-portrait").disabled = true;
      appendLog(`Saved image for ${personName(currentFaceId)} removed · saving paused`);
    } catch (error) {
      appendLog(`Could not remove portrait: ${error.message}`);
    }
  }

  $("consent").addEventListener("change", () => { $("start").disabled = !$("consent").checked || !!media; });
  $("start").addEventListener("click", beginCapture);
  $("stop").addEventListener("click", () => void stopCapture());
  $("voice").addEventListener("click", enrollVoice);
  $("forget-portrait").addEventListener("click", forgetPortrait);
  window.addEventListener("beforeunload", () => { if (media) void stopCapture(false); });

  fetch("/health").then((response) => response.json()).then((health) => {
    const transcript = health.transcriber_backend === "sarvam" ? "SAARAS CODE-MIX" : "LOCAL WHISPER";
    const denoiser = health.speech_enhancer === "deepfilternet" ? " · DEEPFILTER" : "";
    $("provider").textContent = `FACE · ${health.face_backend || "CONFIGURED"} · ${transcript}${denoiser}`;
    $("transcript-mode").textContent = health.transcriber_backend === "sarvam" ? "SARVAM SAARAS · CODE-MIX" : "TRANSCRIBED LOCALLY";
  }).catch(() => { $("provider").textContent = "FACE MODULE · OFFLINE"; });
})();
