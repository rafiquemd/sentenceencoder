package com.corover.sentenceencoder;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;

import javax.annotation.PostConstruct;
import java.nio.charset.StandardCharsets;


@Component
public class Embedding {

	@Value("${app.modelpath}")
	private String modelPath;



	float[][] embed(String[] values) {
		SavedModelBundle savedModelBundle = SavedModelBundle.load(modelPath, "serve");
		byte[][] input = new byte[values.length][];
		for (int i = 0; i < values.length; i++) {
			String val = values[i];
			input[i] = val.getBytes(StandardCharsets.UTF_8);

		}

		Tensor t = Tensor.create(input);

		Tensor result =
				savedModelBundle
						.session()
						.runner()
						.feed("input", t)
						.fetch("output")
						.run()
						.get(0);

		float[][] output = new float[values.length][512];
		result.copyTo(output);

		return output;

	}

}
