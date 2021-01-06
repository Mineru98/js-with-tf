const tf = require("@tensorflow/tfjs-node");

let 온도 = [20,21,22,23];
let 판매량 = [40,42,44,46];

let 원인 = tf.tensor(온도);
let 결과 = tf.tensor(판매량);

let X = tf.input({ shape: [ 1 ] });
let Y = tf.layers.dense({ units: 1}).apply(X);
let model = tf.model({ inputs: X, outputs: Y });
let compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError }
model.compile(compileParam);

let fitParam = { epochs: 10 }
model.fit(원인, 결과,  fitParam).then((result) => {
    let 다음주온도 = [15,16,17,18,19];
    let 다음주원인 = tf.tensor2d(다음주온도, [다음주온도.length, 1]);
    let 다음주결과 = model.predict(다음주원인);
    다음주결과.print()
})