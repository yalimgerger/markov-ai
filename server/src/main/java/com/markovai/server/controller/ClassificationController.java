package com.markovai.server.controller;

import com.markovai.server.ai.ClassificationResult;
import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.DigitMarkovModel;
import com.markovai.server.service.MarkovTrainingService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

//import java.util.Map;

@RestController
public class ClassificationController {

    private static final Logger logger = LoggerFactory.getLogger(ClassificationController.class);
    private final MarkovTrainingService trainingService;

    public ClassificationController(MarkovTrainingService trainingService) {
        this.trainingService = trainingService;
    }

    public static class ClassificationRequest {
        public int[][] pixels;
        // Optional: support flat pixels if needed, but 2D is easier for current setup
    }

    @PostMapping("/classify-digit")
    public ResponseEntity<?> classify(@RequestBody ClassificationRequest request) {
        if (!trainingService.isReady()) {
            return ResponseEntity.status(503).body("Model is still training, please try again later.");
        }

        if (request.pixels == null || request.pixels.length != 28 || request.pixels[0].length != 28) {
            return ResponseEntity.badRequest().body("Invalid image data. Must be 28x28 grayscale pixels.");
        }

        logger.info("Received classification request.");

        // Label -1 because it's unknown
        DigitImage img = new DigitImage(request.pixels, -1);
        DigitMarkovModel model = trainingService.getModel();

        ClassificationResult result = model.classifyWithScores(img);

        return ResponseEntity.ok(result);
    }
}
