#pragma once

#include "Representations/Communication/BHumanMessage.h"
#include "Representations/Communication/GameInfo.h"
#include "Representations/Communication/RobotInfo.h"
#include "Representations/Configuration/DamageConfiguration.h"
#include "Representations/Infrastructure/AudioData.h"
#include "Representations/Infrastructure/FrameInfo.h"
#include "Representations/Modeling/Whistle.h"
//#include "Tools/Debugging/DebugImages.h"
#include "Tools/Module/Module.h"
#include "Tools/RingBuffer.h"
#include "Tools/Streams/Eigen.h"
#include <fftw3.h>


MODULE(WhistleRecognizer,
{,
USES(GameInfo),
    REQUIRES(AudioData),
    REQUIRES(BHumanMessageOutputGenerator),
    REQUIRES(DamageConfigurationHead),
    REQUIRES(FrameInfo),
    REQUIRES(RawGameInfo),
    REQUIRES(RobotInfo),
    PROVIDES(Whistle),
    LOADS_PARAMETERS(
    {,
    (unsigned) bufferSize, /**< The number of samples buffered per channel. */
    (unsigned) sampleRate, /**< The sample rate actually used. */
    (float) minVolume, /**< The minimum volume that must be reached for accepting a whistle [0..1). */
    (int) minAnnotationDelay, /**< The minimum time between annotations announcing a detected whistle. */
    (bool) detectInPlaying, /**< Detect whistles in PLAYING state instead of SET state. */
    (unsigned) minFreq,
    (unsigned) maxFreq,
    (float) maxTimespan,
    (int) samplingRate,
    (float) share, /**< Minimal accepted probability */
    }),
});


class WhistleRecognizer : public WhistleRecognizerBase
{
  std::vector<RingBuffer<AudioData::Sample>> buffers; /**< Sample buffers for all channels. */
  unsigned samplesSize; /**< The number of samples that fft takes */
  bool hasRecorded = false; /**< Was audio recorded in the previous cycle? */
  unsigned sampleIndex = 0; /** Index of next sample to process for subsampling. */
  double* samples; /**< The samples after normalization. */
  fftw_complex* spectrum; /**< The spectrum of the samples. */
  fftw_plan fft; /**< The plan to compute the FFT. */

  /**
   * This method is called when the representation provided needs to be updated.
   * @param theWhistle The representation updated.
   */
  void update(Whistle& theWhistle) override;

  /**
   * This method returns probability of whistle in whole channel
   * @param buffer The audio samples from one channel
   */
  float detect(const RingBuffer<AudioData::Sample>& buffer);

  /**
   * This method returns probability of whistle in selected sample
   * @param bufferedSample The audio sub-samples from one channel
   */
  float detectInSample(const RingBuffer<AudioData::Sample>& bufferedSample);

 public:
  WhistleRecognizer();
  ~WhistleRecognizer();
};
