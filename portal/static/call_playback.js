(function installCallPlayback(root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.OmniCallPlayback = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function callPlaybackFactory() {
  "use strict";

  function supersedeBefore(call, sequence) {
    const turns = [];
    for (const turn of call.turns) {
      if (turn.sequence >= sequence) continue;
      turn.discardReply = true;
      turns.push(turn);
    }
    const playbackInterrupted = Boolean(
      call.playbackTurn && call.playbackTurn.sequence < sequence,
    );
    if (playbackInterrupted) call.playbackTurn = null;
    return { turns, playbackInterrupted };
  }

  function canStart(call, turn) {
    return Boolean(
      !turn.discardReply
      && !call.vadActive
      && turn.sequence === call.nextSequence - 1,
    );
  }

  function retentionState({
    content = "",
    thinking = "",
    toolTrace = [],
    languageSettled = false,
  } = {}) {
    const preserve = Boolean(
      String(content || "").trim()
      || String(thinking || "").trim()
      || (Array.isArray(toolTrace) && toolTrace.length),
    );
    return {
      preserve,
      interrupted: preserve && !languageSettled,
    };
  }

  // Streamed speech is generated in blocks, sometimes slower than it plays
  // (0.8x realtime on a loaded Jetson). Started as soon as the first chunk
  // arrives, each chunk finishes before the next exists and playback
  // stutters. The pacer holds playback until the buffered audio covers the
  // predicted shortfall, estimated duration x (1/rate - 1), using a rate and
  // speaking speed measured on this device and remembered between replies.
  const DEFAULT_PRIOR = Object.freeze({ rate: 1.0, charsPerSecond: 14 });
  const MIN_PREROLL_SECONDS = 0.25;
  const SAFETY = 1.15;
  const RATE_SMOOTHING = 0.5;

  function finiteOr(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? number : fallback;
  }

  function createPacer({ prior = DEFAULT_PRIOR, textChars = 0 } = {}) {
    const rate = finiteOr(prior && prior.rate, DEFAULT_PRIOR.rate);
    const charsPerSecond = finiteOr(
      prior && prior.charsPerSecond, DEFAULT_PRIOR.charsPerSecond,
    );
    const chars = Math.max(0, Number(textChars) || 0);
    return {
      rate,
      charsPerSecond,
      textChars: chars,
      expectedSeconds: chars ? chars / charsPerSecond : 0,
      producedSeconds: 0,
      packets: 0,
      firstArrival: null,
      lastArrival: null,
      firstDuration: 0,
      underruns: 0,
    };
  }

  function measuredRate(pacer) {
    if (pacer.packets < 2 || pacer.lastArrival <= pacer.firstArrival) return pacer.rate;
    // Audio produced after the first packet over the time it took to arrive.
    const sample = (pacer.producedSeconds - pacer.firstDuration)
      / (pacer.lastArrival - pacer.firstArrival);
    return RATE_SMOOTHING * pacer.rate + (1 - RATE_SMOOTHING) * sample;
  }

  function arrive(pacer, durationSeconds, nowSeconds) {
    const duration = Math.max(0, Number(durationSeconds) || 0);
    if (pacer.firstArrival === null) {
      pacer.firstArrival = nowSeconds;
      pacer.firstDuration = duration;
    }
    pacer.lastArrival = nowSeconds;
    pacer.producedSeconds += duration;
    pacer.packets += 1;
  }

  // Seconds of audio to hold before (re)starting playback.
  function requiredLead(pacer) {
    const rate = measuredRate(pacer);
    const remaining = Math.max(0, pacer.expectedSeconds - pacer.producedSeconds);
    const deficit = rate < 1 ? (pacer.expectedSeconds || remaining) * (1 / rate - 1) : 0;
    return Math.max(MIN_PREROLL_SECONDS, deficit * SAFETY * (1 + 0.25 * pacer.underruns));
  }

  function shouldStart(pacer, bufferedSeconds, ended = false) {
    if (ended) return pacer.packets > 0;
    // Without a length estimate, one packet cannot reveal the producer's
    // cadence; wait for the second, as the local speaker does.
    if (!pacer.expectedSeconds) return pacer.packets >= 2;
    return bufferedSeconds >= Math.min(requiredLead(pacer), pacer.expectedSeconds);
  }

  function underrun(pacer) {
    pacer.underruns += 1;
  }

  // Updated per-device estimates to remember after a complete reply.
  function learned(pacer) {
    const rate = measuredRate(pacer);
    const charsPerSecond = pacer.textChars && pacer.producedSeconds > 0.5
      ? RATE_SMOOTHING * pacer.charsPerSecond
        + (1 - RATE_SMOOTHING) * (pacer.textChars / pacer.producedSeconds)
      : pacer.charsPerSecond;
    return { rate, charsPerSecond };
  }

  return {
    supersedeBefore,
    canStart,
    retentionState,
    createPacer,
    arrive,
    requiredLead,
    shouldStart,
    underrun,
    learned,
  };
}));
