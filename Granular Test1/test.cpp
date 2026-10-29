
#include "daisy_pod.h"
#include "daisysp.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

using namespace daisy;
using namespace daisysp;

DaisyPod hw;
volatile bool isRecording = false;
volatile bool isPlaying = false;
const int maxSize = 720000;
volatile int recordIndex = 0;
volatile int startIndex = 0;
volatile int length = 0;
volatile int note = 0;
volatile int recordedLength = 0;
volatile bool generateVoices = false;
volatile int samp = 0;
volatile int mode = 0;
volatile int voiceDensity = 10000;
volatile float sprayRandomness = 0.0f;
volatile float verbDryWet = 0.0f;
volatile float verbTime = 0.0f;
volatile float startPercent;
volatile float endPercent;
constexpr float kKnobChangeThreshold = 0.001f;
float knob1LatchValue = 0.0f;
float knob2LatchValue = 0.0f;
float DSY_SDRAM_BSS bufferL[maxSize];
float DSY_SDRAM_BSS bufferR[maxSize];

volatile float noteMultiplier = 1.0f;

float rmsNormalizerGain = 1.0f;
float rmsAccumulator = 0.0f;
float peakAccumulator = 0.0f;
int rmsSampleCount = 0;
constexpr float kTargetRms = 0.12f;
constexpr float kPeakHeadroom = 0.92f;
constexpr float kMinRmsGain = 0.25f;
constexpr float kMaxRmsGain = 4.0f;
constexpr int kRmsWindowSize = 128;

static inline float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

struct Voice {
    volatile int startIndex;
    volatile int length;
    volatile float currentIndex; // fractional position for resampling
    volatile float speed;        // playback speed (1.0 = original)
    volatile bool active;
    volatile bool stolen;
    volatile float stealStartIndex;
    volatile int stealFadeLen;
};

const int kMaxVoices = 64;
constexpr int kMaxActiveVoices = 32;
const int kVoiceFadeLen = 128;
float voiceMixRamp = 0.0f;
volatile int activeVoiceCount = 0;
Voice voices[kMaxVoices] = {};

constexpr int kReverbNumCombs = 4;
constexpr int kReverbNumAllpasses = 2;
constexpr int kReverbCombDelays[kReverbNumCombs] = {1116, 1188, 1277, 1356};
constexpr int kReverbAllpassDelays[kReverbNumAllpasses] = {556, 441};
float reverbCombBufL[kReverbNumCombs][1600];
float reverbCombBufR[kReverbNumCombs][1600];
int reverbCombIdx[kReverbNumCombs] = {};
float reverbAllpassBufL[kReverbNumAllpasses][600];
float reverbAllpassBufR[kReverbNumAllpasses][600];
int reverbAllpassIdx[kReverbNumAllpasses] = {};

std::mt19937_64 voiceRng;
std::uniform_real_distribution<float> voiceRandomDist(-1.0f, 1.0f);

int createVoice(int index, int len, float speed)
{
    for (int i = 0; i < kMaxVoices; i++)
    {
        if (!voices[i].active)
        {
            float randomnessIndex = voiceRandomDist(voiceRng) * sprayRandomness * static_cast<float>(len);

            voices[i].startIndex = index + static_cast<int>(randomnessIndex);
            voices[i].length = len;
            voices[i].currentIndex = 0.0f;
            voices[i].speed = speed;
            voices[i].active = true;
            voices[i].stolen = false;
            voices[i].stealStartIndex = 0.0f;
            voices[i].stealFadeLen = 0;
            return i;
        }
    }

    // No free voice slots: implement voice stealing.
    // Choose the active voice that is farthest through its playback
    // (highest currentIndex/length progress) to minimize audible artifacts.
    int victim = 0;
    float maxProgress = -1.0f;
    for (int i = 0; i < kMaxVoices; i++)
    {
        if (!voices[i].active)
            continue;

        float progress = 0.0f;
        if (voices[i].length > 0)
            progress = static_cast<float>(voices[i].currentIndex) / static_cast<float>(voices[i].length);

        if (progress > maxProgress)
        {
            maxProgress = progress;
            victim = i;
        }
    }

    // If the chosen voice still has a lot of playback remaining, shorten it
    // so it fades out over the next kVoiceFadeLen samples instead of cutting abruptly.
    int remaining = voices[victim].length - static_cast<int>(voices[victim].currentIndex);
    if (remaining > kVoiceFadeLen)
    {
        if (!voices[victim].stolen)
        {
            voices[victim].stolen = true;
            voices[victim].stealStartIndex = voices[victim].currentIndex;
            voices[victim].stealFadeLen = kVoiceFadeLen;
        }
        return -1;
    }

    // Reuse victim slot for the new voice
    voices[victim].startIndex = index;
    voices[victim].length = len;
    voices[victim].currentIndex = 0.0f;
    voices[victim].speed = speed;
    voices[victim].active = true;
    voices[victim].stolen = false;
    voices[victim].stealStartIndex = 0.0f;
    voices[victim].stealFadeLen = 0;
    return victim;
}

float hannWindow(int n, int len)
{
    if (len <= 1)
        return 1.0f;
    return 0.5f * (1.0f - cosf(3.14159265f * n / (len - 1)));
}

float voiceEnvelope(float currentIndex, int length)
{
    if (length <= 1)
        return 1.0f;

    int fadeLen = length / 2;
    if (fadeLen > 128)
        fadeLen = 128;
    if (fadeLen < 1)
        fadeLen = 1;

    if (currentIndex < fadeLen)
        return hannWindow(static_cast<int>(currentIndex), fadeLen);

    if (currentIndex >= length - fadeLen)
        return hannWindow(static_cast<int>((length - 1) - currentIndex), fadeLen);

    return 1.0f;
}

// Helper: record incoming sample into buffers when recording
static inline void processRecordingSample(float inL, float inR)
{
    if (isRecording)
    {
        if (recordIndex < maxSize)
        {
            bufferL[recordIndex] = inL;
            bufferR[recordIndex] = inR;
            recordIndex++;
            recordedLength = recordIndex;
        }
        else
        {
            isRecording = false;
        }
    }
}

// Mix active voices and update their state. Returns number of active voices and sets voiceL/voiceR.
static inline int mixActiveVoices(float &voiceL, float &voiceR)
{
    float mixL = 0.0f;
    float mixR = 0.0f;
    int activeVoices = 0;

    for (int v = 0; v < kMaxVoices; v++)
    {
        if (!voices[v].active)
            continue;

        // stop when the voice has played its configured length
        if (voices[v].currentIndex >= static_cast<float>(voices[v].length))
        {
            voices[v].active = false;
            continue;
        }

        if (recordedLength <= 0)
        {
            voices[v].active = false;
            continue;
        }

        // fractional position within the source buffer (allow wrap-around)
        float srcPos = static_cast<float>(voices[v].startIndex) + voices[v].currentIndex;
        int idx = static_cast<int>(floorf(srcPos));
        int wrappedIdx = idx % recordedLength;
        if (wrappedIdx < 0)
            wrappedIdx += recordedLength;
        float frac = srcPos - static_cast<float>(idx);
        int nextWrapped = (wrappedIdx + 1) % recordedLength;
        float sampL = bufferL[wrappedIdx] * (1.0f - frac) + bufferL[nextWrapped] * frac;
        float sampR = bufferR[wrappedIdx] * (1.0f - frac) + bufferR[nextWrapped] * frac;

        float env;
        if (voices[v].stolen)
        {
            float fadePos = voices[v].currentIndex - voices[v].stealStartIndex;
            if (fadePos >= static_cast<float>(voices[v].stealFadeLen))
            {
                voices[v].active = false;
                continue;
            }
            float phase = fadePos / static_cast<float>(voices[v].stealFadeLen);
            env = 0.5f * (1.0f + cosf(3.14159265f * phase));
        }
        else
        {
            env = voiceEnvelope(voices[v].currentIndex, voices[v].length);
        }

        mixL += sampL * env;
        mixR += sampR * env;

        // advance fractional index by speed (allows pitch manipulation)
        voices[v].currentIndex += voices[v].speed;
        activeVoices++;

        if (voices[v].currentIndex >= static_cast<float>(voices[v].length))
        {
            voices[v].active = false;
        }
    }

    activeVoiceCount = activeVoices;

    if (activeVoices > 0)
    {
        // Averaging by active voice count makes the output get quieter as density rises.
        // Use a square-root compensation so perceived loudness stays more consistent
        // while still preventing the sum from growing linearly with every added grain.
        float densityComp = 1.0f / std::sqrt(static_cast<float>(activeVoices));
        voiceL = mixL * densityComp;
        voiceR = mixR * densityComp;
        isPlaying = true;
    }
    else
    {
        voiceL = 0.0f;
        voiceR = 0.0f;
        isPlaying = false;
    }

    return activeVoices;
}

static inline void updateVoiceMixRamp(int activeVoices)
{
    float targetMix = activeVoices > 0 ? 1.0f : 0.0f;
    float mixStep = 1.0f / static_cast<float>(kVoiceFadeLen);
    if (voiceMixRamp < targetMix)
        voiceMixRamp = std::min(1.0f, voiceMixRamp + mixStep);
    else if (voiceMixRamp > targetMix)
        voiceMixRamp = std::max(0.0f, voiceMixRamp - mixStep);
}

static inline void maybeGenerateVoicePerBlock(size_t blockSize)
{
    if (!generateVoices || recordedLength <= 0 || isRecording)
        return;

    if (activeVoiceCount >= kMaxActiveVoices)
        return;

    samp += static_cast<int>(blockSize);
    while (samp >= voiceDensity)
    {
        samp -= voiceDensity;

        // compute speed from semitone `note` and global `noteMultiplier`
        float semitoneRatio = powf(2.0f, static_cast<float>(note) / 12.0f);
        float speed = semitoneRatio * noteMultiplier;
        if (createVoice(startIndex, length, speed) < 0)
            break;

        if (activeVoiceCount >= kMaxActiveVoices)
            break;
    }

    if (samp > 479999)
        samp %= 480000;
}

static inline void stopAllVoices()
{
    for (int i = 0; i < kMaxVoices; i++)
    {
        voices[i].active = false;
        voices[i].stolen = false;
        voices[i].stealStartIndex = 0.0f;
        voices[i].stealFadeLen = 0;
        voices[i].currentIndex = 0.0f;
    }

    activeVoiceCount = 0;
    voiceMixRamp = 0.0f;
    isPlaying = false;
}

static inline void initReverb()
{
    for (int i = 0; i < kReverbNumCombs; i++)
    {
        std::fill(&reverbCombBufL[i][0], &reverbCombBufL[i][0] + kReverbCombDelays[i], 0.0f);
        std::fill(&reverbCombBufR[i][0], &reverbCombBufR[i][0] + kReverbCombDelays[i], 0.0f);
        reverbCombIdx[i] = 0;
    }
    for (int i = 0; i < kReverbNumAllpasses; i++)
    {
        std::fill(&reverbAllpassBufL[i][0], &reverbAllpassBufL[i][0] + kReverbAllpassDelays[i], 0.0f);
        std::fill(&reverbAllpassBufR[i][0], &reverbAllpassBufR[i][0] + kReverbAllpassDelays[i], 0.0f);
        reverbAllpassIdx[i] = 0;
    }
}

static inline void processSchroederReverb(float &inOutL, float &inOutR)
{
    if (verbDryWet <= 0.0f)
        return;

    float decay = clampf(verbTime / 10.0f, 0.0f, 1.0f);
    float combFeedback = 0.45f + decay * 0.45f;
    float allpassFeedback = 0.45f + decay * 0.35f;

    float combSumL = 0.0f;
    float combSumR = 0.0f;
    for (int i = 0; i < kReverbNumCombs; i++)
    {
        int idx = reverbCombIdx[i];
        float delayedL = reverbCombBufL[i][idx];
        float delayedR = reverbCombBufR[i][idx];

        reverbCombBufL[i][idx] = inOutL + delayedL * combFeedback;
        reverbCombBufR[i][idx] = inOutR + delayedR * combFeedback;

        combSumL += delayedL;
        combSumR += delayedR;

        idx = (idx + 1) % kReverbCombDelays[i];
        reverbCombIdx[i] = idx;
    }

    float allpassL = combSumL / static_cast<float>(kReverbNumCombs);
    float allpassR = combSumR / static_cast<float>(kReverbNumCombs);
    for (int i = 0; i < kReverbNumAllpasses; i++)
    {
        int idx = reverbAllpassIdx[i];
        float bufOutL = reverbAllpassBufL[i][idx];
        float bufOutR = reverbAllpassBufR[i][idx];

        float outL = bufOutL - allpassFeedback * allpassL;
        float outR = bufOutR - allpassFeedback * allpassR;

        reverbAllpassBufL[i][idx] = allpassL + allpassFeedback * outL;
        reverbAllpassBufR[i][idx] = allpassR + allpassFeedback * outR;

        allpassL = outL;
        allpassR = outR;

        idx = (idx + 1) % kReverbAllpassDelays[i];
        reverbAllpassIdx[i] = idx;
    }

    float wetL = allpassL;
    float wetR = allpassR;
    inOutL = inOutL * (1.0f - verbDryWet) + wetL * verbDryWet;
    inOutR = inOutR * (1.0f - verbDryWet) + wetR * verbDryWet;
}

static inline void applyRmsNormalization(float &outL, float &outR)
{
    float powerL = outL * outL;
    float powerR = outR * outR;
    float samplePower = 0.5f * (powerL + powerR);
    float samplePeak = std::max(std::fabs(outL), std::fabs(outR));

    rmsAccumulator += samplePower;
    peakAccumulator = std::max(peakAccumulator, samplePeak);
    rmsSampleCount++;

    if (rmsSampleCount >= kRmsWindowSize)
    {
        float rms = std::sqrt(rmsAccumulator / static_cast<float>(rmsSampleCount));
        float desiredGain = kTargetRms / std::max(rms, 1e-6f);
        if (peakAccumulator > 1e-6f)
            desiredGain = std::min(desiredGain, kPeakHeadroom / peakAccumulator);
        desiredGain = clampf(desiredGain, kMinRmsGain, kMaxRmsGain);

        constexpr float kMaxGainStep = 0.0025f;
        float deltaGain = desiredGain - rmsNormalizerGain;
        if (std::fabs(deltaGain) > kMaxGainStep)
            deltaGain = (deltaGain > 0.0f) ? kMaxGainStep : -kMaxGainStep;
        rmsNormalizerGain += deltaGain;

        rmsAccumulator = 0.0f;
        peakAccumulator = 0.0f;
        rmsSampleCount = 0;
    }

    // React immediately to peaks between RMS windows so gain changes cannot
    // drive a transient into clipping.
    if (samplePeak > 1e-6f)
        rmsNormalizerGain = std::min(rmsNormalizerGain, kPeakHeadroom / samplePeak);

    outL *= rmsNormalizerGain;
    outR *= rmsNormalizerGain;
}


void AudioCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size)
{
    maybeGenerateVoicePerBlock(size);

    for (size_t i = 0; i < size; i++)
    {
        float inL = in[0][i];
        float inR = in[1][i];

        // Recording input if enabled
        processRecordingSample(inL, inR);

        // Mix active voices
        float voiceL = 0.0f, voiceR = 0.0f;
        int activeVoices = mixActiveVoices(voiceL, voiceR);

        // Smoothly crossfade between dry and granular voices
        updateVoiceMixRamp(activeVoices);

        // Combine dry input and voice mix
        float outL = inL * (1.0f - voiceMixRamp) + voiceL * voiceMixRamp;
        float outR = inR * (1.0f - voiceMixRamp) + voiceR * voiceMixRamp;

        // Apply reverb only when granular voices are currently active
        if (activeVoices > 0)
        {
            processSchroederReverb(outL, outR);
        }

        // Normalize both granular output and dry input passthrough.
        applyRmsNormalization(outL, outR);

        out[0][i] = outL;
        out[1][i] = outR;
    }
}

// --- Main-loop helpers ---
static inline void processControlsAndEncoder()
{
    hw.ProcessAnalogControls();
    hw.ProcessDigitalControls();
    note += hw.encoder.Increment();
}

static inline void handleButton1()
{
    if (hw.button1.RisingEdge() && !isPlaying)
    {
        if (isRecording)
        {
            isRecording = false;
        }
        else
        {
            recordIndex = 0;
            recordedLength = 0;
            isRecording = true;
            isPlaying = false;
        }
    }
}

static inline void handleModeAndKnobs()
{
    if (hw.encoder.RisingEdge())
    {
        mode = (mode+1)%3;
        knob1LatchValue = hw.knob1.Value();
        knob2LatchValue = hw.knob2.Value();
    }

    const float knob1Value = hw.knob1.Value();
    const float knob2Value = hw.knob2.Value();

    if (mode == 0)
    {
        if (std::fabs(knob1Value - knob1LatchValue) > kKnobChangeThreshold)
        {
            startPercent = knob1Value;
            knob1LatchValue = knob1Value;
        }

        if (std::fabs(knob2Value - knob2LatchValue) > kKnobChangeThreshold)
        {
            endPercent = knob2Value;
            knob2LatchValue = knob2Value;
        }
    }
    else if (mode == 1)
    {
        if (std::fabs(knob1Value - knob1LatchValue) > kKnobChangeThreshold)
        {
            voiceDensity = static_cast<int>(knob1Value * 20000.0f + 2000.0f);
            knob1LatchValue = knob1Value;
        }

        if (std::fabs(knob2Value - knob2LatchValue) > kKnobChangeThreshold)
        {
            sprayRandomness = knob2Value;
            knob2LatchValue = knob2Value;
        }
    }else if (mode == 2){
        if (std::fabs(knob1Value - knob1LatchValue) > kKnobChangeThreshold)
        {
            verbDryWet = knob1Value;
            knob1LatchValue = knob1Value;
        }

        if (std::fabs(knob2Value - knob2LatchValue) > kKnobChangeThreshold)
        {
            verbTime = knob2Value*10.0f;
            knob2LatchValue = knob2Value;
        }
    }
}

static inline void computeStartAndLength()
{
    if (recordedLength > 0)
    {
        startIndex = static_cast<int>(startPercent * (recordedLength - 1));
        length = static_cast<int>(endPercent * (recordedLength - 1));
        if (startIndex < 0)
            startIndex = 0;
        if (length <= 0)
            length = 1;
    }
}

static inline void handleButton2()
{
    if (hw.button2.RisingEdge() && !isRecording)
    {
        const bool wasGenerating = generateVoices;
        generateVoices = !generateVoices;

        if (wasGenerating && !generateVoices)
        {
            stopAllVoices();
        }
    }
}

static inline void updateLedsFromState()
{
    if (isRecording)
    {
        hw.led1.Set(1.0f, 0.0f, 0.0f);
    }
    else if (isPlaying)
    {
        hw.led1.Set(0.0f, 1.0f, 0.0f);
    }
    else
    {
        hw.led1.Set(0.0f, 0.0f, 0.0f);
    }

    static const float Colors[12][3] = {
        {1.0f, 0.0f, 0.0f}, // Red
        {1.0f, 0.5f, 0.0f}, // Orange
        {1.0f, 1.0f, 0.0f}, // Yellow
        {0.0f, 1.0f, 0.0f}, // Green
        {0.0f, 1.0f, 1.0f}, // Cyan
        {0.0f, 0.0f, 1.0f}, // Blue
        {1.0f, 0.0f, 1.0f}, // Magenta
        {1.0f, 1.0f, 1.0f}, // White
        {0.5f, 0.5f, 0.5f}, // Gray
        {0.5f, 0.0f, 0.0f}, // Dark Red
        {0.5f, 0.5f, 0.0f}, // Olive
        {0.0f, 0.5f, 0.5f}  // Teal
    };
    // int colorIndex = std::abs(note % 12);
    int colorIndex = mode;
    hw.led2.Set(Colors[colorIndex][0], Colors[colorIndex][1], Colors[colorIndex][2]);
    hw.UpdateLeds();
}

int main(void)
{
    hw.Init();
    std::random_device rd;
    voiceRng.seed(rd());
    initReverb();
    // hw.seed.StartLog(true);
    hw.SetAudioBlockSize(48);
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);
    hw.StartAdc();
    hw.StartAudio(AudioCallback);

    while (1)
    {
        processControlsAndEncoder();
        handleButton1();
        handleModeAndKnobs();
        computeStartAndLength();
        handleButton2();
        updateLedsFromState();
    }
}
