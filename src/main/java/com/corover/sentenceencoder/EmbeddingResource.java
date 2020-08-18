package com.corover.sentenceencoder;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

@RestController
public class EmbeddingResource {

	@Autowired
	private Embedding embedding;

	@GetMapping("/encoder")
	public ResponseEntity<?> getEmbedding(@RequestParam(value = "input") String input) {
		String[] s = {input};
		float[][] output = embedding.embed(s);
		Map<String, float[][]> outputMap = new HashMap<>();
		outputMap.put("output", output);
		return ResponseEntity.status(HttpStatus.OK).body(outputMap);
	}
}
