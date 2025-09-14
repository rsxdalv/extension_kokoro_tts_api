let audioContext;
let audioWorkletNode;
let openAIApiKey = "sk-111111";

function parseWavHeader(buffer) {
  const view = new DataView(buffer);
  // Sample rate is at bytes 24-27 (little endian)
  const sampleRate = view.getUint32(24, true);
  // Number of channels is at bytes 22-23 (little endian)
  const channels = view.getUint16(22, true);
  // Bits per sample is at bytes 34-35 (little endian)
  const bitsPerSample = view.getUint16(34, true);

  return { sampleRate, channels, bitsPerSample };
}

async function initAudioWorklet(wavSampleRate) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: wavSampleRate,
  });

  // Simple AudioWorklet processor for PCM streaming
  const processorCode = `
        class PCMProcessor extends AudioWorkletProcessor {
            constructor() {
                super();
                this.buffer = [];
                this.pendingBytes = new Uint8Array(0); // Buffer for incomplete samples
                this.port.onmessage = (event) => {
                    if (event.data.pcmData) {
                        // Combine any pending bytes with new data
                        const newData = new Uint8Array(event.data.pcmData);
                        const combined = new Uint8Array(this.pendingBytes.length + newData.length);
                        combined.set(this.pendingBytes);
                        combined.set(newData, this.pendingBytes.length);
                        
                        // Calculate how many complete 16-bit samples we have
                        const completeSamples = Math.floor(combined.length / 2);
                        const bytesToProcess = completeSamples * 2;
                        
                        if (completeSamples > 0) {
                            // Process complete samples
                            const int16Array = new Int16Array(combined.buffer.slice(0, bytesToProcess));
                            const float32Data = new Float32Array(int16Array.length);
                            for (let i = 0; i < int16Array.length; i++) {
                                float32Data[i] = int16Array[i] / 32768.0; // Convert 16-bit to float
                            }
                            // Use a loop instead of spread operator to avoid call stack overflow
                            for (let i = 0; i < float32Data.length; i++) {
                                this.buffer.push(float32Data[i]);
                            }
                        }
                        
                        // Store any remaining incomplete bytes
                        if (combined.length > bytesToProcess) {
                            this.pendingBytes = combined.slice(bytesToProcess);
                        } else {
                            this.pendingBytes = new Uint8Array(0);
                        }
                    }
                };
            }
            
            process(inputs, outputs, parameters) {
                const output = outputs[0];
                if (output.length > 0 && this.buffer.length > 0) {
                    const channelData = output[0];
                    for (let i = 0; i < channelData.length && this.buffer.length > 0; i++) {
                        channelData[i] = this.buffer.shift() || 0;
                    }
                }
                return true;
            }
        }
        registerProcessor('pcm-processor', PCMProcessor);
    `;

  const blob = new Blob([processorCode], { type: "application/javascript" });
  const processorUrl = URL.createObjectURL(blob);

  await audioContext.audioWorklet.addModule(processorUrl);
  audioWorkletNode = new AudioWorkletNode(audioContext, "pcm-processor");
  audioWorkletNode.connect(audioContext.destination);

  URL.revokeObjectURL(processorUrl);
}

async function streamWavAudio() {
  const rs = await fetch("http://127.0.0.1:7778/v1/audio/speech", {
    method: "POST",
    headers: {
      Authorization: "Bearer " + openAIApiKey,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input: "They have spines They have spines spines spines",
      model: "chatterbox",
      response_format: "wav",
      voice: "random",
      params: {
        desired_length: 5,
        max_length: 20,
      },
    }),
  }).then((res) => res.body);

  const reader = rs.getReader();
  let headerParsed = false;
  let wavInfo = null;

  reader.read().then(function process({ done, value }) {
    if (done) {
      return;
    }

    if (!headerParsed) {
      // Parse WAV header to get sample rate
      wavInfo = parseWavHeader(value.buffer);
      console.log("WAV Info:", wavInfo);

      // Initialize AudioWorklet with correct sample rate
      initAudioWorklet(wavInfo.sampleRate).then(() => {
        // Skip WAV header (first 44 bytes typically)
        const pcmData = value.slice(44);
        audioWorkletNode.port.postMessage({ pcmData });
        headerParsed = true;
        reader.read().then(process);
      });
      return;
    }

    // Send PCM data to AudioWorklet
    audioWorkletNode.port.postMessage({ pcmData: value });
    reader.read().then(process);
  });
}

// Start streaming
streamWavAudio();
