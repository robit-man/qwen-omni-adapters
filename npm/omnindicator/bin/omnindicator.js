#!/usr/bin/env node

"use strict";

const { main } = require("../lib/cli");

main(process.argv.slice(2)).catch((error) => {
  const message = error && error.message ? error.message : String(error);
  process.stderr.write(`omnindicator: ${message}\n`);
  process.exitCode = 1;
});
