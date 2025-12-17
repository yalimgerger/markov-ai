package com.markovai.server.ai.inference;

import com.markovai.server.ai.ClassificationResult;
import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.MarkovFieldDigitClassifier;

public class LayeredInferenceEngine implements InferenceEngine {

    private final MarkovFieldDigitClassifier classifier;

    public LayeredInferenceEngine(MarkovFieldDigitClassifier classifier) {
        this.classifier = classifier;
    }

    @Override
    public InferenceResult infer(DigitImage img) {
        // Delegate to existing classifier logic
        ClassificationResult legacyResult = classifier.classifyWithDetails(img);

        return new InferenceResult(
                legacyResult.getPredictedDigit(),
                legacyResult.getLogLikelihoods(),
                null, // belief not used yet
                1, // always 1 iteration for layered
                "layered");
    }
}
