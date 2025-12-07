package com.markovai.server.controller;

import com.markovai.server.ai.ClassificationResult;
import com.markovai.server.ai.DigitImage;
import com.markovai.server.service.MarkovTrainingService;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
public class ClassificationController {

    // private static final Logger logger =
    // LoggerFactory.getLogger(ClassificationController.class);
    private final MarkovTrainingService trainingService;

    public ClassificationController(MarkovTrainingService trainingService) {
        this.trainingService = trainingService;
    }

    public static class ClassificationRequest {
        public int[][] pixels;
        // Optional: support flat pixels if needed, but 2D is easier for current setup
    }

    @PostMapping("/classify-digit")
    public ClassificationResult classifyDigit(@RequestBody Map<String, int[][]> payload) {
        if (!trainingService.isReady()) {
            throw new RuntimeException("Model is not ready yet.");
        }

        int[][] pixels = payload.get("pixels");
        if (pixels == null || pixels.length != 28 || pixels[0].length != 28) { // Added check for pixels[0].length
            throw new IllegalArgumentException("Invalid pixel data. Must be 28x28 grayscale pixels.");
        }

        DigitImage img = new DigitImage(pixels, -1);
        return trainingService.getModel().classifyWithScores(img);
    }
}
