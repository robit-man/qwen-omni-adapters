import assertModule from "assert";
import { createRequire } from "module";

const assert = assertModule.strict;
const require = createRequire(import.meta.url);
require("./static/session_cache.js");

const { createSessionCache, memoryStorage } = globalThis.OmniSessionCacheFactory;

async function main() {
  let now = 10_000;
  const storage = memoryStorage();
  const cache = createSessionCache({ storage, now: () => now });
  const scope = "test-session-scope";

  await cache.save(scope, {
    history: [{ role: "user", content: "remember this" }],
    messages: [{ role: "user", content: "remember this", media: [{ kind: "video", data: "AAAA" }] }],
  });
  assert.equal((await cache.load(scope)).history[0].content, "remember this");

  now += 120_000;
  await cache.markLeft(scope);
  now += 10 * 365 * 24 * 60 * 60 * 1000;
  assert.equal((await cache.load(scope)).messages[0].media[0].kind, "video");
  assert.equal(cache.retention, "explicit-clear");

  await cache.save(scope, { history: [{ role: "assistant", content: "new" }] });
  await cache.clear(scope);
  assert.equal(await cache.load(scope), null);

  console.log(JSON.stringify({ status: "passed", retention: cache.retention }));
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
