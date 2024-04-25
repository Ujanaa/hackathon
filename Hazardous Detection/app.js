const express = require("express");
const cors = require('cors');
const fs = require("fs");
const client = require("https");
const request = require("request");
const tf = require("@tensorflow/tfjs-node");

const app = express();

var corsOptions = {
  origin: 'https://ganesh-dagadi.github.io',
  optionsSuccessStatus: 200
}

app.use(cors(corsOptions));
app.options('*', cors(corsOptions))

const model_url = "file://jsmodel.tfjs/model.json"; //path to the model
const api_key = process.env.MapsAPI; //Bing Maps API key

app.get("/api/:lat/:lon/:zoom", async (req, res) => {
  try {
    if (global.gc) {global.gc();}
  } catch (e) {
    console.log(e);
  }
  //loading model
  console.log("Loading model");
  let model = await tf.loadLayersModel(model_url);
  console.log("Loaded model");

  //downloading satellite image
  const { lat, lon, zoom } = req.params;
  const imageUrl = `https://dev.virtualearth.net/REST/V1/Imagery/Map/Aerial/${lat}%2C${lon}/${zoom}?mapSize=224%2C224&format=jpeg&key=${api_key}`;
  const imagePth = "./pic.jpeg";
  
  console.log("Downloading satellite image:");
  console.log(imageUrl);
  await client.get(imageUrl, (res) => {
    res.pipe(fs.createWriteStream(imagePth));
  });
  await new Promise(resolve => setTimeout(resolve, 5000));
  console.log("Downloaded satellite image");

  //predictions on the image
  console.log("Reading satellite image");
  const buf = fs.readFileSync(imagePth);
  const input = tf.node.decodeJpeg(buf);
  let imageTensor = tf.node.decodeJpeg(buf);
  imageTensor = imageTensor.expandDims(0);
  console.log(`Success: converted local file to a ${imageTensor.shape} tensor`);
  const pred = await model.predict(imageTensor, { batchSize: 4 }).data();
  model = null;
  res.json({
    data: pred
  });
});

app.listen(5000, () => {
  console.log("Server Listening");
});
