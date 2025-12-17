package com.markovai.server.ai.inference;

import com.markovai.server.ai.MarkovFieldDigitClassifier;
import com.markovai.server.ai.hierarchy.FactorGraphBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class InferenceEngineFactory {

    private static final Logger logger = LoggerFactory.getLogger(InferenceEngineFactory.class);

    public static InferenceEngine create(FactorGraphBuilder.ConfigRoot config, MarkovFieldDigitClassifier classifier) {
        String topology = (config != null && config.topology != null) ? config.topology : "layered";
        return create(topology, classifier);
    }

    public static InferenceEngine create(String topology, MarkovFieldDigitClassifier classifier) {
        // Default to layered if missing or invalid
        if (topology == null || topology.trim().isEmpty()) {
            logger.warn("Topology not specified, defaulting to 'layered'");
            topology = "layered";
        }

        LayeredInferenceEngine layered = new LayeredInferenceEngine(classifier);

        switch (topology.toLowerCase()) {
            case "layered":
                return layered;
            case "network":
                return new NetworkInferenceEngine(layered);
            default:
                logger.warn("Unknown topology '{}', defaulting to 'layered'", topology);
                return layered;
        }
    }
}
