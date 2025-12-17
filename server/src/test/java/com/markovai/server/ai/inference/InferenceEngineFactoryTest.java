package com.markovai.server.ai.inference;

import com.markovai.server.ai.MarkovFieldDigitClassifier;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class InferenceEngineFactoryTest {

    @Test
    public void testFactoryCreation() {
        // Pass null classifier as we only check the returned type
        MarkovFieldDigitClassifier mockClassifier = null;

        // 1. Default (null arg)
        InferenceEngine engine = InferenceEngineFactory.create((String) null, mockClassifier);
        assertTrue(engine instanceof LayeredInferenceEngine, "Should default to Layered");
        // infer will throw NPE if called?
        // Wait, LayeredE calls classifier.classify.
        // I cannot call infer with null classifier.
        // But the factory return checks are safe.

        // 2. Explicit Layered
        engine = InferenceEngineFactory.create("layered", mockClassifier);
        assertTrue(engine instanceof LayeredInferenceEngine);

        // 3. Network
        engine = InferenceEngineFactory.create("network", mockClassifier);
        assertTrue(engine instanceof NetworkInferenceEngine);

        // 4. Invalid -> Default
        engine = InferenceEngineFactory.create("foobar", mockClassifier);
        assertTrue(engine instanceof LayeredInferenceEngine, "Unknown topology should fallback to Layered");
    }
}
