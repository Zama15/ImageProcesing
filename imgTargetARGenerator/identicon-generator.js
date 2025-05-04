const jdenticon = require("jdenticon");
const fs = require("fs");
const path = require("path");

const SIZE = 200;
const FILEPATH = process.argv[2];

if (!FILEPATH) {
  console.error("Please provide a path to the text file.");
  process.exit(1);
}

try {
  const input = fs.readFileSync(path.resolve(FILEPATH), "utf-8").trim();
  const png = jdenticon.toPng(input, SIZE);

  const outputDir = path.join(__dirname, "img");
  if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir);

  const outputFilename = path.join(
    outputDir,
    `identicon-${path.basename(FILEPATH, path.extname(FILEPATH))}.png`
  );
  fs.writeFileSync(outputFilename, png);

  console.log(`Generated ${outputFilename}`);
} catch (error) {
  console.error("Error reading file or generating image:", error.message);
}
