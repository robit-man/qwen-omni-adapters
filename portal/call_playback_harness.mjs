#!/usr/bin/env node

import assertModule from "assert";
import { createRequire } from "module";

const assert = assertModule.strict;
const require = createRequire(import.meta.url);
const playback = require("./static/call_playback.js");

// A 10 s reply generated at 0.8x realtime needs ~2.5 s of lead to never
// starve; the pacer holds at least that much (with margin) before playing.
const slow = playback.createPacer({ prior: { rate: 0.8, charsPerSecond: 14 }, textChars: 140 });
assert.equal(slow.expectedSeconds, 10);
const lead = playback.requiredLead(slow);
assert.ok(lead >= 2.5 && lead < 3.5, `lead ${lead}`);
assert.equal(playback.shouldStart(slow, 1.0), false);
assert.equal(playback.shouldStart(slow, lead), true);

// Faster-than-realtime generation plays after a short preroll.
const fast = playback.createPacer({ prior: { rate: 1.6, charsPerSecond: 14 }, textChars: 140 });
assert.equal(playback.requiredLead(fast), 0.25);

// The rate is measured from arrivals, not assumed: 0.64 s chunks every 0.8 s.
const measured = playback.createPacer({ prior: { rate: 1.0 }, textChars: 140 });
for (let index = 0; index < 6; index += 1) playback.arrive(measured, 0.64, index * 0.8);
assert.ok(playback.learned(measured).rate < 0.95, "measured slow producer");
assert.ok(playback.requiredLead(measured) > 0.25);

// Without a length estimate, wait for the second packet as the speaker does.
const unknown = playback.createPacer({});
playback.arrive(unknown, 0.64, 0);
assert.equal(playback.shouldStart(unknown, 0.64), false);
playback.arrive(unknown, 0.64, 0.5);
assert.equal(playback.shouldStart(unknown, 1.28), true);

// A stall raises the lead for the rest of the reply; end-of-stream plays out.
const before = playback.requiredLead(slow);
playback.underrun(slow);
assert.ok(playback.requiredLead(slow) > before);
assert.equal(playback.shouldStart(unknown, 0.1, true), true);

// Speaking speed is learned for the next reply's duration estimate.
const speech = playback.createPacer({ prior: { rate: 1, charsPerSecond: 14 }, textChars: 200 });
playback.arrive(speech, 5, 0);
playback.arrive(speech, 5, 4);
assert.ok(playback.learned(speech).charsPerSecond > 14);
