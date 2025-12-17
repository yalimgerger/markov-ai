package com.markovai.server.ai.inference;

import com.markovai.server.ai.DigitImage;

public interface InferenceEngine {
    /**
     * Infer the digit for the given image using the configured topology.
     */
    InferenceResult infer(DigitImage img);
}
