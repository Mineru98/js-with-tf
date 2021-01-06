const tf = require('@tensorflow/tfjs-node');
async function start() {
	const model = await tf.loadLayersModel("https://bg24-front.run.goorm.io/data/model.json");
	var 다음주온도 = [15, 16, 17, 18, 19];
	model.predict(tf.tensor(다음주온도)).print();
	
}

start();
