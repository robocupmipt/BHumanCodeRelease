#include "WhistleRecognizer.h"
#include "Platform/SystemCall.h"
#include "Platform/Thread.h"
#include "Tools/Debugging/Annotation.h"
#include "Tools/Debugging/DebugDrawings.h"
#include "Tools/Math/BHMath.h"
#include <algorithm>
#include <limits>
#include <type_traits>


MAKE_MODULE(WhistleRecognizer, modeling)

static DECLARE_SYNC;


WhistleRecognizer::WhistleRecognizer() {
  // Allocate memory for FFTW plans
  samplesSize = unsigned(maxTimespan * samplingRate);
  samples = fftw_alloc_real(samplesSize * 2);
  std::memset(samples, 0, sizeof(double) * samplesSize * 2);
  spectrum = fftw_alloc_complex(samplesSize + 1);

  // Create FFTW plans
  SYNC;
  fft = fftw_plan_dft_r2c_1d(int(samplesSize) * 2, samples, spectrum, FFTW_MEASURE);
}

WhistleRecognizer::~WhistleRecognizer() {
  SYNC;
  fftw_destroy_plan(fft);
  fftw_free(spectrum);
  fftw_free(samples);
}

void WhistleRecognizer::update(Whistle &theWhistle) {
  // Empty buffers when entering a state where it should be recorded.
  const bool shouldRecord = ((!detectInPlaying &&
      theRawGameInfo.state == STATE_SET
      && theGameInfo.state != STATE_PLAYING)
      || (detectInPlaying
          && theGameInfo.state == STATE_PLAYING
          && theRobotInfo.penalty == PENALTY_NONE))
      && !SystemCall::soundIsPlaying();

  if (!hasRecorded && shouldRecord) {
    buffers.clear();
  }
  hasRecorded = shouldRecord;
  // Adapt number of channels to audio data.
  buffers.resize(theAudioData.channels);
  for (auto &buffer : buffers) {
    buffer.reserve(bufferSize);
  }

  // Append current samples to buffers and sample down if necessary
  ASSERT(theAudioData.sampleRate % sampleRate == 0);
  const unsigned stepSize = theAudioData.sampleRate / sampleRate * theAudioData.channels;
  for(; sampleIndex < theAudioData.samples.size(); sampleIndex += stepSize)
  {
    for(unsigned channel = 0; channel < theAudioData.channels; ++channel) {
      buffers[channel].push_front(theAudioData.samples[sampleIndex + channel]);
    }
  }
  sampleIndex -= theAudioData.samples.size();

  // No whistles can be detected while sound is playing.
  if(SystemCall::soundIsPlaying()) {
    theWhistle.channelsUsedForWhistleDetection = 0;
  }

  // Count number of channels if they were set to zero and no sound is playing.
  if(!theWhistle.channelsUsedForWhistleDetection && !SystemCall::soundIsPlaying()) {
    for (unsigned i = 0; i < buffers.size(); ++i) {
      if (!theDamageConfigurationHead.audioChannelsDefect[i]) {
        ++theWhistle.channelsUsedForWhistleDetection;
      }
    }
  }

  // Compute first channel index to access damage configuration.
  const int firstBuffer = theDamageConfigurationHead.audioChannelsDefect[0] ? 1 : 0;

  DEBUG_RESPONSE_ONCE("module:WhistleRecognizer:debug") {
    if(buffers[firstBuffer].full()) {
      float correlation = detect(buffers[firstBuffer]);
      OUTPUT_TEXT("Recorded whistle with correlation = " << correlation);
    } else {
      OUTPUT_TEXT("Buffer is not full. Actual size is " +  std::to_string(buffers[firstBuffer].size()));
      OUTPUT_TEXT("theAudioData.samples.size():");
      OUTPUT_TEXT(std::to_string(theAudioData.samples.size()));
      OUTPUT_TEXT("theAudioData.channels:");
      OUTPUT_TEXT(std::to_string(theAudioData.channels));
      for(unsigned channel = 0; channel < theAudioData.channels; ++channel) {
        OUTPUT_TEXT("buffers[channel].size() with channel = " + std::to_string(channel));
        OUTPUT_TEXT(std::to_string(buffers[channel].size()));
      }
    }
  }

  for (unsigned i = 0; i < buffers.size(); i++) {
    if (theDamageConfigurationHead.audioChannelsDefect[i]) {
      continue;
    }
    float correlation = detect(buffers[i]);
    if (correlation > share) {
      if (theFrameInfo.getTimeSince(theWhistle.lastTimeWhistleDetected) > minAnnotationDelay) {
        ANNOTATION("WhistleRecognizerBase",
                   "Detect whistle with correlation " << static_cast<int>(correlation * 100.f) << "%");
        theWhistle.lastTimeWhistleDetected = theFrameInfo.time;
      }
    }
  }
}

float WhistleRecognizer::detect(const RingBuffer<AudioData::Sample> &buffer) {
  float maxCorrelation = 0;
  RingBuffer<AudioData::Sample> bufferedSample;
  bufferedSample.reserve(samplesSize);

  for (unsigned i = 0; i < buffer.size(); i++) {
    if (bufferedSample.size() < samplesSize) {
      bufferedSample.push_front(buffer[i]);
      continue;
    }
    float correlation = detectInSample(bufferedSample);
    maxCorrelation = std::max(maxCorrelation, correlation);
    bufferedSample.clear();
  }
  return maxCorrelation;
}

float WhistleRecognizer::detectInSample(const RingBuffer<AudioData::Sample> &bufferedSample) {
  // Compute volume of samples.
  float volume = 0;
  for (AudioData::Sample sample : bufferedSample) {
    volume = std::max(volume, std::abs(static_cast<float>(sample)));
  }

  // Abort if not loud enough.
  if (volume < minVolume) {
    return 0.f;
  }
  // Copy samples to FFTW input and normalize them.
  const double factor = 1.0 / volume;
  for (unsigned i = 0; i < bufferedSample.size(); ++i) {
    samples[i] = bufferedSample[i] * factor;
  }
  // samples -> spectrum
  fftw_execute(fft);

  float total_sum = 0;
  float best_sum = 0;
  for (unsigned i = 0; i < samplesSize; ++i) {
    total_sum += sqr(spectrum[i][0]) + sqr(spectrum[i][1]);
    if (i < maxFreq and i > minFreq) {
      best_sum += sqr(spectrum[i][0]) + sqr(spectrum[i][1]);
    }

  }

  return best_sum / total_sum;
}
