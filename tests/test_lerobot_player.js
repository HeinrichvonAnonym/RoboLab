/* Pure playback/edit checks, runnable even without a browser. */
const assert = require("assert");
const {mergeIntervals, isExcluded, keptSummary, playbackIndex} = require("../scripts/lerobot_tools/static/app.js");
assert.deepStrictEqual(mergeIntervals([[4, 6], [1, 3], [2, 5], [9, 10]]), [[1, 6], [9, 10]]);
assert.strictEqual(isExcluded(.2, {exclude: [[.2, .4]]}), true);
assert.strictEqual(isExcluded(.4, {exclude: [[.2, .4]]}), true);
assert.strictEqual(isExcluded(.1, {exclude: [[.2, .4]]}), false);
assert.strictEqual(isExcluded(.1, {drop: true}), true);
assert.deepStrictEqual(keptSummary([0, 1, 2, 3, 4, 5], {exclude: [[2, 3]]}), {count: 4, episodes: 2});
assert.deepStrictEqual(keptSummary([0, 1], {drop: true}), {count: 0, episodes: 0});
assert.strictEqual(playbackIndex(0, 1000, 15, .25, 100), 3);
assert.strictEqual(playbackIndex(0, 1000, 15, 4, 100), 60);
assert.strictEqual(playbackIndex(10, 1000, 15, 4, 20), 20);
assert.strictEqual(playbackIndex(10, 0, 15, 1, 20), 10);
console.log("Playback, inclusive cuts, merge and episode checks passed.");
