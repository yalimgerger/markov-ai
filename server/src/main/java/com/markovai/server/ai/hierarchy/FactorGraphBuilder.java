package com.markovai.server.ai.hierarchy;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.markovai.db.DigitImageDao;
import com.markovai.db.MarkovChainResultDao;
import com.markovai.db.SqliteInitializer;
import com.markovai.server.ai.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FactorGraphBuilder {

    private static final Logger logger = LoggerFactory.getLogger(FactorGraphBuilder.class);
    // DB_PATH is now injected
    private static final String CHAIN_VERSION = "v1";

    // Dependencies to inject into leaf nodes
    private final DigitMarkovModel rowModel;
    private final DigitMarkovModel colModel;
    private final DigitMarkovModel patchModel;

    private final MultiSequenceExtractor rowExtractor;
    private final MultiSequenceExtractor colExtractor;
    private final SequenceExtractor patchExtractor;
    private final DigitPatch4x4UnigramModel patch4x4Model;

    private final DigitImageDao imageDao;
    private final MarkovChainResultDao resultDao;

    public FactorGraphBuilder(DigitMarkovModel rowModel, DigitMarkovModel colModel, DigitMarkovModel patchModel,
            MultiSequenceExtractor rowExtractor, MultiSequenceExtractor colExtractor,
            SequenceExtractor patchExtractor, DigitPatch4x4UnigramModel patch4x4Model, String dbPath) {
        this.rowModel = rowModel;
        this.colModel = colModel;
        this.patchModel = patchModel;
        this.rowExtractor = rowExtractor;
        this.colExtractor = colExtractor;
        this.patchExtractor = patchExtractor;
        this.patch4x4Model = patch4x4Model;

        // Initialize DB
        try {
            SqliteInitializer.initialize(dbPath);
            logger.info("Initialized SQLite cache at {}", dbPath);
        } catch (SQLException e) {
            logger.error("Failed to initialize SQLite", e);
            throw new RuntimeException(e);
        }
        this.imageDao = new DigitImageDao(dbPath);
        this.resultDao = new MarkovChainResultDao(dbPath);
    }

    public static class ConfigNode {
        public String id;
        public String type;
        public List<String> children;
        public Map<String, Double> weights;
        public Double smoothingLambda;
        public Patch4x4FeedbackConfig feedback;
    }

    public static class NetworkConfig {
        public Boolean enabled;
        public Integer maxIters;
        public Double temperature;
        public Double priorWeight;
        public Double damping;
        public Double stopEpsilon;
        public Double epsilon;
        public Boolean debugStats;
    }

    public static class PayoffConfig {
        public Boolean enabled;
        public String scheme;
        public Double confStrong;
        public Double confWeak;
        public Double scaleStrong;
        public Double scaleWeakCorrect;
        public Double scaleWeakIncorrect;
        public Boolean requireConvergence;
        public Integer maxItersForConverged;
        public Boolean applyToCorrect;
        public Boolean applyToIncorrect;
    }

    public static class BasinConfig {
        public Boolean enabled;
        public String scheme;
        public String rivalSelection;
        public Boolean useTrajectory;
        public Integer trajectoryMinStep;
        public Double winnerReinforce;
        public Double rivalPenalize;
        public String deltaWeightMode;
        public Double minDeltaGate;
        public Boolean normalizeTrajectory;
        public Double penalizeCap;
    }

    public static class ObserverWeightsConfig {
        public Boolean enabled;
        public Double alpha;
        public Double temperature;
        public String scaleK; // "NUM_OBSERVERS" or a double string
        public Boolean usePayoffScale;
        public Boolean updateOnlyIfIncorrect;
        public Boolean requireConvergence;
        public String advantageMode; // "signed_margin" or "scaled_margin"
        public Double marginClip;
        public Integer logEveryN;

        // Cross-Entropy Fields
        public String updateRule; // "heuristic" (default) or "cross_entropy"
        public Double scoreSoftmaxTemperature;
        public Boolean centerObserverScores;
        public String centerMode;

        // Standardization & Regularization
        public Boolean standardizeObserverScores;
        public Double l2;
    }

    public static class LearningConfig {
        public ObserverWeightsConfig observerWeights;
        public PayoffConfig payoff;
        public BasinConfig basin;
    }

    public static class ConfigRoot {
        public String topology = "layered";
        public String markov_data_directory;
        public NetworkConfig network;
        public LearningConfig learning;
        public List<ConfigNode> nodes;
        public String rootNodeId;
    }

    public Map<String, DigitFactorNode> build(InputStream jsonStream) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            ConfigRoot config = mapper.readValue(jsonStream, ConfigRoot.class);

            Map<String, DigitFactorNode> nodes = new HashMap<>();

            // 1. Create Nodes
            for (ConfigNode cn : config.nodes) {
                DigitFactorNode node = null;
                switch (cn.type) {
                    case "RowMarkovNode":
                        RowMarkovEvaluator rowEval = new RowMarkovEvaluator(rowModel, rowExtractor, CHAIN_VERSION);
                        CachedMarkovChainEvaluator cachedRow = new CachedMarkovChainEvaluator(rowEval, imageDao,
                                resultDao);
                        Patch4x4FeedbackConfig rowFeedback = cn.feedback != null ? cn.feedback
                                : Patch4x4FeedbackConfig.disabled();
                        node = new RowMarkovNode(cn.id, cachedRow, rowExtractor, rowFeedback);
                        break;
                    case "ColumnMarkovNode":
                        ColumnMarkovEvaluator colEval = new ColumnMarkovEvaluator(colModel, colExtractor,
                                CHAIN_VERSION);
                        CachedMarkovChainEvaluator cachedCol = new CachedMarkovChainEvaluator(colEval, imageDao,
                                resultDao);
                        Patch4x4FeedbackConfig colFeedback = cn.feedback != null ? cn.feedback
                                : Patch4x4FeedbackConfig.disabled();
                        node = new ColumnMarkovNode(cn.id, cachedCol, colExtractor, colFeedback);
                        break;
                    case "PatchMarkovNode":
                        Patch2x2Evaluator patchEval = new Patch2x2Evaluator(patchModel, patchExtractor, CHAIN_VERSION);
                        CachedMarkovChainEvaluator cachedPatch = new CachedMarkovChainEvaluator(patchEval, imageDao,
                                resultDao);
                        node = new PatchMarkovNode(cn.id, cachedPatch);
                        break;
                    case "Patch4x4Node":
                        double lambda = cn.smoothingLambda != null ? cn.smoothingLambda : 0.0;
                        Patch4x4FeedbackConfig feedback = cn.feedback != null ? cn.feedback
                                : Patch4x4FeedbackConfig.disabled();
                        // Note: Bypassing CachedMarkovChainEvaluator for online learning node
                        node = new Patch4x4Node(cn.id, patch4x4Model, lambda, feedback);
                        break;
                    case "WeightedSumNode":
                        // Children wired later
                        WeightedSumNode wsNode = new WeightedSumNode(cn.id, new ArrayList<>(),
                                cn.weights != null ? cn.weights : new HashMap<>());
                        // Check if standardization is enabled in global config
                        if (config.learning != null && config.learning.observerWeights != null) {
                            boolean owEnabled = Boolean.TRUE.equals(config.learning.observerWeights.enabled);
                            boolean standardize = Boolean.TRUE
                                    .equals(config.learning.observerWeights.standardizeObserverScores);
                            wsNode.setStandardizeObserverScores(owEnabled && standardize);
                        }
                        node = wsNode;
                        break;
                    default:
                        logger.warn("Unknown node type: {}", cn.type);
                }
                if (node != null) {
                    nodes.put(cn.id, node);
                }
            }

            // 2. Wire Children
            for (ConfigNode cn : config.nodes) {
                if (cn.children != null && !cn.children.isEmpty() && "WeightedSumNode".equals(cn.type)) {
                    WeightedSumNode parent = (WeightedSumNode) nodes.get(cn.id);
                    if (parent != null) {
                        for (String childId : cn.children) {
                            DigitFactorNode child = nodes.get(childId);
                            if (child != null) {
                                parent.addChild(child);
                            } else {
                                logger.error("Missing child node id: {} for parent {}", childId, cn.id);
                            }
                        }
                    }
                }
            }

            logger.info("Factor Graph built with {} nodes. Root: {}", nodes.size(), config.rootNodeId);
            return nodes;

        } catch (Exception e) {
            throw new RuntimeException("Failed to build factor graph from JSON", e);
        }
    }
}
