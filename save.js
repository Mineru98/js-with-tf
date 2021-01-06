const tf = require('@tensorflow/tfjs-node');

// 1. 과거의 데이터를 준비합니다.
var 온도 = [20, 21, 22, 23];
var 판매량 = [40, 42, 44, 46];
var 원인 = tf.tensor(온도);
var 결과 = tf.tensor(판매량);

// 2. 모델의 모양을 만듭니다.
var X = tf.input({ shape: [1] });
var Y = tf.layers.dense({ units: 1 }).apply(X);
var model = tf.model({ inputs: X, outputs: Y });
var compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError };
model.compile(compileParam);

// 3. 데이터로 모델을 학습시킵니다.
var fitParam = {
	epochs: 4000,
	callbacks: {
		onEpochEnd: function (epoch, logs) {
			// Math.sqrt 는 제곱근이다. 제곱한 뒤 그것의 루트 연산한 결과 값
			// 따라서 RMSE는 loss가 평균이므로 loss에 제곱을 한 뒤 루트를 씌운 결과가 되어야 하니 해당 식이 맞음.
			console.log('epoch', epoch, logs, "RMSE=>", Math.sqrt(logs.loss));
		},
	},
}; // loss 추가 예제
model.fit(원인, 결과, fitParam).then(function (result) {
	// 4. 모델을 이용합니다.
	// 4.1 기존의 데이터를 이용
	var 예측한결과 = model.predict(원인);
	예측한결과.print();
	model.save('file:///workspace/js-with-tf/data');
});

// 4.2 새로운 데이터를 이용
// var 다음주온도 = [15, 16, 17, 18, 19];
// var 다음주원인 = tf.tensor(다음주온도);
// var 다음주결과 = model.predict(다음주원인);
// 다음주결과.print();