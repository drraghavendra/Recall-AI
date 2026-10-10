(() => {
  const TARGET_RATE = 16000;
  const FRAME_PERIOD_MS = 5000;
  const AUDIO_CHUNK_SAMPLES = 1600; // 100 ms at 16 kHz
  const $ = (id) => document.getElementById(id);

  const video = $("video");
  const canvas = document.createElement("canvas");
  const context2d = canvas.getContext("2d", { willReadFrequently: false });
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
  let voiceParts = [];
  let voiceTimer = null;

  const setStatus = (text, kind = "") => {
    $("connection").textContent = text;
    $("connection").className = `connection ${kind}`;
  };

  const appendLog = (text) => {
    $("event-log").textContent = `${new Date().toLocaleTimeString()} · ${text}`;
  };

  function bytesToBase64(bytes) {
    let binary = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
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
    const outputLength = Math.floor((input.length * TARGET_RATE) / inputRate);
    const output = new Float32Array(outputLength);
    const step = inputRate / TARGET_RATE;
    for (let i = 0; i < outputLength; i += 1) {
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
      const chunk = joined.subarray(offset, offset + AUDIO_CHUNK_SAMPLES);
      sendAudio(pcm16(chunk));
      offset += AUDIO_CHUNK_SAMPLES;
    }
    pendingAudio = joined.slice(offset);
    event.outputBuffer.getChannelData(0).fill(0);
  }

  function sendFrame() {
    if (!socket || socket.readyState !== WebSocket.OPEN || video.readyState < 2 || !video.videoWidth) return;
    if (socket.bufferedAmount > 4_000_000) {
      appendLog("Skipped a frame because the connection is busy.");
      return;
    }
    const scale = Math.min(1, 640 / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    context2d.drawImage(video, 0, 0, canvas.width, canvas.height);
    const jpeg = canvas.toDataURL("image/jpeg", 0.78).split(",")[1];
    socket.send(JSON.stringify({ type: "frames", ts: Date.now() / 1000, frames: [jpeg] }));
  }

  async function fetchPersona(faceId) {
    try {
      const response = await fetch(`/persona/${encodeURIComponent(faceId)}`);
      if (response.status === 404) {
        $("persona").innerHTML = '<p class="first-meeting">You’re meeting them for the first time.</p><p>No persona has been created yet.</p>';
        return;
      }
      if (!response.ok) throw new Error(`Persona request returned ${response.status}`);
      renderPersona(await response.json());
    } catch (error) {
      $("persona").textContent = `Could not load persona: ${error.message}`;
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

  function handleEvent(event) {
    if (event.type === "face_update") {
      if (event.face_id !== currentFaceId) {
        currentFaceId = event.face_id;
        if (currentFaceId) {
          $("face-name").textContent = `Face ${currentFaceId.slice(0, 8)}`;
          $("persona").textContent = "Looking up saved details…";
          void fetchPersona(currentFaceId);
        } else {
          $("face-name").textContent = "No face in view";
          $("persona").textContent = "No one is currently in focus.";
        }
      }
      appendLog(event.face_id ? `Face match ${event.face_id.slice(0, 8)} · confidence ${Math.round(event.confidence * 100)}%` : "No face found in focus.");
    } else if (event.type === "persona_updated") {
      if (event.face_id === currentFaceId) renderPersona(event.persona);
      appendLog("Persona updated.");
    } else if (event.type === "conversation_started") {
      $("talking").textContent = "TALKING";
      $("talking").classList.add("active");
      appendLog("Conversation started.");
    } else if (event.type === "conversation_ended") {
      $("talking").textContent = "IDLE";
      $("talking").classList.remove("active");
      appendLog("Conversation ended.");
    }
  }

  async function beginCapture() {
    if (!$("consent").checked) return;
    $("start").disabled = true;
    setStatus("Requesting camera and microphone…");
    try {
      media = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      video.srcObject = media;
      await video.play();
      $("placeholder").classList.add("hidden");

      socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
      socket.onmessage = (message) => {
        try { handleEvent(JSON.parse(message.data)); } catch (error) { appendLog(`Invalid server event: ${error.message}`); }
      };
      socket.onclose = () => setStatus("Disconnected", "error");
      socket.onerror = () => setStatus("WebSocket connection failed", "error");
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
      frameTimer = setInterval(sendFrame, FRAME_PERIOD_MS);
      $("stop").disabled = false;
      $("voice").disabled = false;
      setStatus("Connected", "live");
      appendLog("Camera and microphone are streaming to the backend.");
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
      setStatus("Not connected");
      appendLog("Capture stopped.");
    }
  }

  async function enrollVoice() {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    voiceParts = [];
    $("voice").disabled = true;
    $("voice").textContent = "Recording… 10 sec";
    appendLog("Speak naturally for 10 seconds to enroll your own voice.");
    voiceTimer = setTimeout(async () => {
      voiceTimer = null;
      const total = voiceParts.reduce((sum, part) => sum + part.length, 0);
      const audio = new Uint8Array(total);
      let offset = 0;
      for (const part of voiceParts) { audio.set(part, offset); offset += part.length; }
      voiceParts = null;
      $("voice").textContent = "Enroll my voice · 10 sec";
      try {
        const response = await fetch("/self-voice", {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: audio,
        });
        if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
        appendLog("Voice profile saved. Your speech can now be excluded from other personas.");
      } catch (error) {
        appendLog(`Voice enrollment failed: ${error.message}`);
      } finally {
        $("voice").disabled = !socket || socket.readyState !== WebSocket.OPEN;
      }
    }, 10_000);
  }

  $("consent").addEventListener("change", () => { $("start").disabled = !$("consent").checked || !!media; });
  $("start").addEventListener("click", beginCapture);
  $("stop").addEventListener("click", () => void stopCapture());
  $("voice").addEventListener("click", enrollVoice);
  window.addEventListener("beforeunload", () => { if (media) void stopCapture(false); });

  fetch("/health").then((response) => response.json()).then((health) => {
    const provider = health.face_backend || "configured";
    $("provider").textContent = `Face module · ${provider}`;
  }).catch(() => { $("provider").textContent = "Face module · unavailable"; });
})();
