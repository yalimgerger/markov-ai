package com.markovai.server.ai.inference;

import com.markovai.server.ai.DigitImage;

public class NetworkInferenceEngine implements InferenceEngine {

    private final LayeredInferenceEngine layeredEngine;

    public NetworkInferenceEngine(LayeredInferenceEngine layeredEngine) {
        this.layeredEngine = layeredEngine;
    }

    @Override
    public InferenceResult infer(DigitImage img) {
        // No-op for now: delegate strictly to layered engine
        InferenceResult layeredResult = layeredEngine.infer(img);

        return new InferenceResult(
                layeredResult.getPredictedDigit(),
                layeredResult.getScores(),
                layeredResult.getBelief(),
                1,
                "network" // But report as network topology
        );
    }
}
