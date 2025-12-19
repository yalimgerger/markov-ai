package com.markovai.server.ai.inference;

import com.markovai.server.ai.MarkovFieldDigitClassifier;
import com.markovai.server.ai.hierarchy.FactorGraphBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class InferenceEngineFactory {

    private static final Logger logger = LoggerFactory.getLogger(InferenceEngineFactory.class);

    public static InferenceEngine create(FactorGraphBuilder.ConfigRoot config, MarkovFieldDigitClassifier classifier) {
        String topology = (config != null && config.topology != null) ? config.topology : "layered";

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
                // If config is null, pass null - logic handles default fallback
                return new NetworkInferenceEngine(classifier, config);
            default:
                logger.warn("Unknown topology '{}', defaulting to 'layered'", topology);
                return layered;
        }
    }

    // Deprecated or testing overload
    public static InferenceEngine create(String topology, MarkovFieldDigitClassifier classifier) {
        // Just create a dummy config wrapper
        FactorGraphBuilder.ConfigRoot cfg = new FactorGraphBuilder.ConfigRoot();
        cfg.topology = topology;
        return create(cfg, classifier);
    }
}
