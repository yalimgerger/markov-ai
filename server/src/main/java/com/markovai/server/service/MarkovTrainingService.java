package com.markovai.server.service;

import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.RowColumnDigitClassifier;
import com.markovai.server.ai.MarkovFieldDigitClassifier;
import com.markovai.server.ai.inference.InferenceEngine;
import com.markovai.server.ai.inference.InferenceEngineFactory;
import com.markovai.server.ai.hierarchy.DigitFactorNode;
import com.markovai.server.ai.hierarchy.FactorGraphBuilder;
import com.markovai.server.util.DataPathResolver;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;
import com.markovai.server.ai.learning.PayoffCalculator;

import javax.imageio.ImageIO;
import jakarta.annotation.PostConstruct;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
public class MarkovTrainingService {

    private static final Logger logger = LoggerFactory.getLogger(MarkovTrainingService.class);

    @org.springframework.beans.factory.annotation.Autowired
    private org.springframework.context.ConfigurableApplicationContext context;

    @org.springframework.beans.factory.annotation.Autowired
    private org.springframework.boot.ApplicationArguments appArgs;

    private final RowColumnDigitClassifier model = new RowColumnDigitClassifier();
    private final com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model = new com.markovai.server.ai.DigitPatch4x4UnigramModel();
    private com.markovai.server.ai.DigitGradientUnigramModel gradModel;
    private boolean isReady = false;

    public RowColumnDigitClassifier getModel() {
        return model;
    }

    public boolean isReady() {
        return isReady;
    }

    @PostConstruct
    public void init() {
        new Thread(() -> {
            try {
                logger.info("Initializing Markov Training Service...");
                String dataDir = DataPathResolver.resolveDataDirectory();
                List<DigitImage> trainingData = loadImages("file:" + dataDir + "/mnist/training/*/*.png");
                List<DigitImage> testingData = loadImages("file:" + dataDir + "/mnist/testing/*/*.png");

                logger.info("Dataset loaded: trainExamples={}, testExamples={}", trainingData.size(),
                        testingData.size());

                if (trainingData.size() != 60000) {
                    logger.warn("Training dataset size ({}) deviates from expected 60000", trainingData.size());
                }
                if (testingData.size() != 10000) {
                    logger.warn("Testing dataset size ({}) deviates from expected 10000", testingData.size());
                }

                if (trainingData.isEmpty()) {
                    logger.error("No training data found!");
                    return;
                }

                model.train(trainingData);

                // Train 4x4 Unigram Model
                logger.info("Training 4x4 Patch Model...");
                for (DigitImage img : trainingData) {
                    int[][] binary = com.markovai.server.ai.DigitMarkovModel.binarize(img.pixels, 128);
                    patch4x4Model.trainOnImage(img.label, binary);
                }
                patch4x4Model.finalizeProbabilities();
                logger.info("4x4 Patch Model Trained.");

                // Train Gradient Model
                try {
                    ObjectMapper mapper = new ObjectMapper();
                    FactorGraphBuilder.ConfigRoot config = mapper.readValue(
                            getClass().getResourceAsStream("/mrf_config.json"),
                            FactorGraphBuilder.ConfigRoot.class);

                    FactorGraphBuilder.ConfigNode gradNode = null;
                    if (config.nodes != null) {
                        for (FactorGraphBuilder.ConfigNode n : config.nodes) {
                            if ("gradient_orientation_unigram".equals(n.type)) {
                                gradNode = n;
                                break;
                            }
                        }
                    }

                    if (gradNode != null) {
                        int bs = gradNode.blockSize != null ? gradNode.blockSize : 4;
                        int nb = gradNode.numBins != null ? gradNode.numBins : 8;
                        int fb = gradNode.flatBin != null ? gradNode.flatBin : 8;
                        double mt = gradNode.magThreshold != null ? gradNode.magThreshold : 50.0;
                        double sa = gradNode.smoothingAlpha != null ? gradNode.smoothingAlpha : 1.0;

                        gradModel = new com.markovai.server.ai.DigitGradientUnigramModel(bs, nb, fb, mt, sa);
                        gradModel.train(trainingData);
                    } else {
                        // Default fallback if not in config yet (so we can pass non-null if needed)
                        // But if not in config, we might pass null. FactorGraphBuilder handles null
                        // gracefully?
                        // FGB logs warning if type is requested but model is null.
                        // We'll create a default one anyway just in case config adds it later without
                        // code change?
                        // No, better to stick to config. If missing, gradModel remains null.
                        logger.info("No gradient_orientation_unigram node found in config, skipping training.");
                    }
                } catch (Exception e) {
                    logger.error("Failed to init gradient model", e);
                }

                if (!testingData.isEmpty()) {
                    boolean runVerification = "true".equalsIgnoreCase(System.getProperty("verifyFeedbackNoLeakage"))
                            || (appArgs != null && appArgs.containsOption("verifyFeedbackNoLeakage") && "true"
                                    .equalsIgnoreCase(appArgs.getOptionValues("verifyFeedbackNoLeakage").get(0)));

                    boolean runSweep = "true".equalsIgnoreCase(System.getProperty("verifyFeedbackSweep"))
                            || (appArgs != null && appArgs.containsOption("verifyFeedbackSweep")
                                    && "true".equalsIgnoreCase(appArgs.getOptionValues("verifyFeedbackSweep").get(0)));

                    boolean runMultiSeed = "true"
                            .equalsIgnoreCase(System.getProperty("verifyFeedbackNoLeakageMultiSeed"))
                            || (appArgs != null && appArgs.containsOption("verifyFeedbackNoLeakageMultiSeed") && "true"
                                    .equalsIgnoreCase(
                                            appArgs.getOptionValues("verifyFeedbackNoLeakageMultiSeed").get(0)));

                    boolean runAdaptSweep = "true"
                            .equalsIgnoreCase(System.getProperty("verifyFeedbackNoLeakageAdaptSweep"))
                            || (appArgs != null && appArgs.containsOption("verifyFeedbackNoLeakageAdaptSweep")
                                    && "true".equalsIgnoreCase(
                                            appArgs.getOptionValues("verifyFeedbackNoLeakageAdaptSweep").get(0)));

                    boolean runNetworkSanity = "true"
                            .equalsIgnoreCase(System.getProperty("networkAttractorSanity"))
                            || (appArgs != null && appArgs.containsOption("networkAttractorSanity")
                                    && "true".equalsIgnoreCase(
                                            appArgs.getOptionValues("networkAttractorSanity").get(0)));

                    if (runNetworkSanity) {
                        runNetworkAttractorSanityCheck(model, testingData, patch4x4Model, gradModel);
                    } else if ("true".equalsIgnoreCase(System.getProperty("networkConvergenceSweep"))
                            || (appArgs != null && appArgs.containsOption("networkConvergenceSweep")
                                    && "true".equalsIgnoreCase(
                                            appArgs.getOptionValues("networkConvergenceSweep").get(0)))) {
                        runNetworkConvergenceSweep(model, testingData, patch4x4Model, gradModel);
                    } else if (runAdaptSweep) {
                        runAdaptationSizeSweep(model, trainingData, testingData, patch4x4Model, gradModel);
                    } else if (runSweep) {
                        runFeedbackSweep(model, trainingData, testingData, patch4x4Model, gradModel);
                    } else if (runMultiSeed) {
                        boolean useRowFeedback = false;
                        boolean useColFeedback = false;
                        String mode = System.getProperty("feedbackMode");
                        if (mode == null && appArgs != null && appArgs.containsOption("feedbackMode")) {
                            List<String> values = appArgs.getOptionValues("feedbackMode");
                            if (values != null && !values.isEmpty()) {
                                mode = values.get(0);
                            }
                        }

                        if (mode != null && !mode.isEmpty()) {
                            switch (mode.toUpperCase()) {
                                case "PATCH":
                                    useRowFeedback = false;
                                    useColFeedback = false;
                                    break;
                                case "PATCH_ROW":
                                    useRowFeedback = true;
                                    useColFeedback = false;
                                    break;
                                case "PATCH_COL":
                                    useRowFeedback = false;
                                    useColFeedback = true;
                                    break;
                                case "PATCH_ROW_COL":
                                    useRowFeedback = true;
                                    useColFeedback = true;
                                    break;
                                default:
                                    logger.warn("Unknown feedbackMode '{}', defaulting to PATCH", mode);
                                    useRowFeedback = false;
                                    useColFeedback = false;
                            }
                            logger.info("Feedback configuration set via feedbackMode={}: row={}, col={}", mode,
                                    useRowFeedback, useColFeedback);
                        } else {
                            // Fallback to legacy flags
                            useRowFeedback = "true".equalsIgnoreCase(System.getProperty("rowFeedback"))
                                    || (appArgs != null && appArgs.containsOption("rowFeedback") && "true"
                                            .equalsIgnoreCase(appArgs.getOptionValues("rowFeedback").get(0)));
                            useColFeedback = "true".equalsIgnoreCase(System.getProperty("colFeedback"))
                                    || (appArgs != null && appArgs.containsOption("colFeedback") && "true"
                                            .equalsIgnoreCase(appArgs.getOptionValues("colFeedback").get(0)));
                        }

                        runLeakageFreeMultiSeedVerification(model, testingData, trainingData, patch4x4Model, gradModel,
                                useRowFeedback, useColFeedback);
                    } else if (runVerification) {
                        boolean useRowFeedback = false;
                        boolean useColFeedback = false;
                        String mode = System.getProperty("feedbackMode");
                        if (mode == null && appArgs != null && appArgs.containsOption("feedbackMode")) {
                            List<String> values = appArgs.getOptionValues("feedbackMode");
                            if (values != null && !values.isEmpty()) {
                                mode = values.get(0);
                            }
                        }

                        if (mode != null && !mode.isEmpty()) {
                            switch (mode.toUpperCase()) {
                                case "PATCH":
                                    useRowFeedback = false;
                                    useColFeedback = false;
                                    break;
                                case "PATCH_ROW":
                                    useRowFeedback = true;
                                    useColFeedback = false;
                                    break;
                                case "PATCH_COL":
                                    useRowFeedback = false;
                                    useColFeedback = true;
                                    break;
                                case "PATCH_ROW_COL":
                                    useRowFeedback = true;
                                    useColFeedback = true;
                                    break;
                                default:
                                    useRowFeedback = false;
                                    useColFeedback = false;
                            }
                        } else {
                            useRowFeedback = "true".equalsIgnoreCase(System.getProperty("rowFeedback"))
                                    || (appArgs != null && appArgs.containsOption("rowFeedback") && "true"
                                            .equalsIgnoreCase(appArgs.getOptionValues("rowFeedback").get(0)));
                            useColFeedback = "true".equalsIgnoreCase(System.getProperty("colFeedback"))
                                    || (appArgs != null && appArgs.containsOption("colFeedback") && "true"
                                            .equalsIgnoreCase(appArgs.getOptionValues("colFeedback").get(0)));
                        }

                        runLeakageFreeVerification(model, testingData, trainingData, patch4x4Model, gradModel,
                                useRowFeedback,
                                useColFeedback);
                    } else {
                        // Legacy Evaluation
                        model.evaluateAccuracy(testingData);

                        // MRF Evaluation
                        try (InputStream is = getClass().getResourceAsStream("/mrf_config.json")) {
                            if (is != null) {
                                FactorGraphBuilder builder = new FactorGraphBuilder(
                                        model.getRowModel(), model.getColumnModel(), model.getPatchModel(),
                                        model.getRowExtractor(), model.getColumnExtractor(), model.getPatchExtractor(),
                                        patch4x4Model, gradModel, DataPathResolver.resolveDbPath());

                                ObjectMapper mapper = new ObjectMapper();
                                FactorGraphBuilder.ConfigRoot config = mapper.readValue(
                                        getClass().getResourceAsStream("/mrf_config.json"),
                                        FactorGraphBuilder.ConfigRoot.class);

                                logger.info("Topology={} (default layered if missing)",
                                        (config.topology != null ? config.topology : "layered"));

                                Map<String, DigitFactorNode> nodes = builder
                                        .build(getClass().getResourceAsStream("/mrf_config.json"));
                                DigitFactorNode root = nodes.get(config.rootNodeId);

                                if (root != null) {
                                    MarkovFieldDigitClassifier mrf = new MarkovFieldDigitClassifier(root);
                                    InferenceEngine engine = InferenceEngineFactory.create(config, mrf);
                                    // Default evaluation on test set (treat as test)
                                    mrf.evaluateAccuracy(testingData, true, engine);
                                } else {
                                    logger.error("MRF Root node not found: {}", config.rootNodeId);
                                }
                            } else {
                                logger.error("mrf_config.json not found!");
                            }
                        } catch (Exception ex) {
                            logger.error("Failed to init MRF", ex);
                        }
                    }
                }

                isReady = true;
                logger.info("Markov Model is ready for classification.");
            } catch (Exception e) {
                logger.error("Failed to initialize Markov model", e);
            }
        }).start();
    }

    private List<DigitImage> loadImages(String pattern) throws IOException {
        List<DigitImage> images = new ArrayList<>();
        java.util.Set<String> loadedHashes = new java.util.HashSet<>();

        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        String searchPattern = (pattern != null) ? pattern : "";
        Resource[] resources = resolver.getResources(searchPattern);

        logger.info("Found {} image resources for pattern: {}", resources.length, pattern);

        for (Resource res : resources) {
            try (InputStream is = res.getInputStream()) {
                BufferedImage bi = ImageIO.read(is);
                if (bi == null)
                    continue;

                int width = bi.getWidth();
                int height = bi.getHeight();
                int[][] pixels = new int[height][width];

                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        // Get grayscale value. PNGs are likely grayscale, but safely extract brightness
                        int clr = bi.getRGB(x, y);
                        int red = (clr & 0x00ff0000) >> 16;
                        int green = (clr & 0x0000ff00) >> 8;
                        int blue = clr & 0x000000ff;
                        // Simple average for grayscale
                        int gray = (red + green + blue) / 3;
                        pixels[y][x] = gray;
                    }
                }

                String hash = computeHash(pixels);
                // Deduplicate based on content hash
                if (hash != null && loadedHashes.contains(hash)) {
                    continue;
                }
                if (hash != null) {
                    loadedHashes.add(hash);
                }

                // Extract label from parent directory name
                // Format: .../mnist/training/5/img.png -> label is 5
                int label = -1;
                String relPath = null;
                try {
                    String path = res.getURL().getPath();
                    String[] parts = path.split("/");

                    int idx = -1;
                    for (int i = 0; i < parts.length; i++) {
                        if ("training".equals(parts[i]) || "testing".equals(parts[i])) {
                            idx = i;
                            break;
                        }
                    }

                    if (idx != -1 && idx < parts.length - 1) {
                        StringBuilder sb = new StringBuilder();
                        for (int i = idx; i < parts.length; i++) {
                            if (sb.length() > 0)
                                sb.append("/");
                            sb.append(parts[i]);
                        }
                        relPath = sb.toString();

                        // Label is parts[length-2]
                        label = Integer.parseInt(parts[parts.length - 2]);
                    } else {
                        // Fallback
                        label = Integer.parseInt(parts[parts.length - 2]);
                        relPath = "unknown/" + label + "/" + parts[parts.length - 1];
                    }

                } catch (Exception e) {
                    logger.warn("Could not determine metadata for {}", res.getDescription());
                }

                images.add(new DigitImage(pixels, label, relPath, hash));
            } catch (Exception e) {
                logger.warn("Failed to load image: {}", res.getFilename(), e);
            }
        }
        return images;
    }

    private String computeHash(int[][] pixels) {
        try {
            // Binarize using same logic as main model
            int[][] binary = com.markovai.server.ai.DigitMarkovModel.binarize(pixels, 128);
            java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
            for (int r = 0; r < 28; r++) {
                for (int c = 0; c < 28; c++) {
                    digest.update((byte) binary[r][c]);
                }
            }
            byte[] hash = digest.digest();
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1)
                    hexString.append('0');
                hexString.append(hex);
            }
            return hexString.toString();
        } catch (Exception e) {
            return null;
        }
    }

    private static class BasinMetrics {
        int count = 0;
        double sumReinforceScale = 0.0;
        double sumPenalizeScale = 0.0;
        double sumPosDw = 0.0;
        double sumPosDr = 0.0;
        double sumWReinforce = 0.0;
        double sumWPenalize = 0.0;
        double sumPayoffScale = 0.0;
        int countUniformFallback = 0;
        int countCorrectAttractor = 0;
        int countIncorrectAttractor = 0;
        List<Integer> iterations = new ArrayList<>();
        long sumIters = 0;

        void update(double reinforce, double penalize, boolean fallback, double posDw, double posDr,
                boolean correctAttractor, int iters, double wReinforce, double wPenalize, double payoffScale) {
            count++;
            sumReinforceScale += reinforce;
            sumPenalizeScale += penalize;
            sumPosDw += posDw;
            sumPosDr += posDr;
            sumWReinforce += wReinforce;
            sumWPenalize += wPenalize;
            sumPayoffScale += payoffScale;
            if (fallback)
                countUniformFallback++;
            if (correctAttractor)
                countCorrectAttractor++;
            else
                countIncorrectAttractor++;
            iterations.add(iters);
            sumIters += iters;
        }

        void printSummary(Logger logger) {
            if (count > 0) {
                String msg = String.format(
                        "Basin Metrics (N=%d): MeanPayoff=%.4f MeanReinforce=%.4f MeanPenalize=%.4f " +
                                "MeanWeights(W=%.4f, R=%.4f) Fallback=%.1f%% CorrectAttractor=%.1f%% MeanIters=%.1f",
                        count,
                        sumPayoffScale / count,
                        sumReinforceScale / count,
                        sumPenalizeScale / count,
                        sumWReinforce / count,
                        sumWPenalize / count,
                        (countUniformFallback * 100.0) / count,
                        (countCorrectAttractor * 100.0) / count,
                        (double) sumIters / count);
                logger.info(msg);
            } else {
                logger.info("Basin Metrics: No updates recorded (N=0)");
            }
            // Reset
            count = 0;
            sumReinforceScale = 0;
            sumPenalizeScale = 0;
            sumPosDw = 0;
            sumPosDr = 0;
            sumWReinforce = 0;
            sumWPenalize = 0;
            sumPayoffScale = 0;
            countUniformFallback = 0;
            countCorrectAttractor = 0;
            countIncorrectAttractor = 0;
            iterations.clear();
            sumIters = 0;
        }
    }

    private static class LeakageFreeResult {
        long seed;
        double baselineAcc;
        double frozenAcc;

        public LeakageFreeResult(long seed, double baselineAcc, double frozenAcc) {
            this.seed = seed;
            this.baselineAcc = baselineAcc;
            this.frozenAcc = frozenAcc;
        }

        public double getDelta() {
            return frozenAcc - baselineAcc;
        }

    }

    private void runLeakageFreeMultiSeedVerification(RowColumnDigitClassifier model, List<DigitImage> testData,
            List<DigitImage> trainData, com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model,
            com.markovai.server.ai.DigitGradientUnigramModel gradModel,
            boolean useRowFeedback, boolean useColFeedback) {
        logger.info("============================================================");
        logger.info("STARTING MULTI-SEED LEAKAGE-FREE VERIFICATION");
        logger.info("============================================================");

        try {
            // Parse seeds
            String seedsStr = System.getProperty("adaptSeeds");
            long[] seeds;
            if (seedsStr != null && !seedsStr.isEmpty()) {
                String[] parts = seedsStr.split(",");
                seeds = new long[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    seeds[i] = Long.parseLong(parts[i].trim());
                }
            } else {
                seeds = new long[] { 12345L, 22222L, 33333L, 44444L, 55555L };
            }

            List<LeakageFreeResult> results = new ArrayList<>();
            // Compute baseline once to be efficient? No, strict requirement is full reset.
            // We will run the full protocol for each seed.

            for (long seed : seeds) {
                logger.info("Running protocol for seed={}, rowFeedback={}, colFeedback={}", seed, useRowFeedback,
                        useColFeedback);
                LeakageFreeResult result = performLeakageFreeProtocol(model, testData, trainData, patch4x4Model,
                        gradModel, seed,
                        false, useRowFeedback, useColFeedback, 2000, null);
                results.add(result);
                logger.info("Seed={}  Baseline={}  Frozen={}  Delta={}",
                        seed, String.format("%.4f", result.baselineAcc), String.format("%.4f", result.frozenAcc),
                        String.format("%+.4f", result.getDelta()));
            }

            // Statistics
            double sumFrozen = 0;
            double sumDelta = 0;
            double minDelta = Double.MAX_VALUE;
            double maxDelta = -Double.MAX_VALUE;

            for (LeakageFreeResult r : results) {
                sumFrozen += r.frozenAcc;
                double d = r.getDelta();
                sumDelta += d;
                if (d < minDelta)
                    minDelta = d;
                if (d > maxDelta)
                    maxDelta = d;
            }

            double meanFrozen = sumFrozen / results.size();
            double meanDelta = sumDelta / results.size();

            double varFrozen = 0;
            double varDelta = 0;
            for (LeakageFreeResult r : results) {
                varFrozen += Math.pow(r.frozenAcc - meanFrozen, 2);
                varDelta += Math.pow(r.getDelta() - meanDelta, 2);
            }
            double stdFrozen = (results.size() > 1) ? Math.sqrt(varFrozen / (results.size() - 1)) : 0.0;
            double stdDelta = (results.size() > 1) ? Math.sqrt(varDelta / (results.size() - 1)) : 0.0;

            String modeName = "PATCH";
            if (useRowFeedback && useColFeedback)
                modeName = "PATCH_ROW_COL";
            else if (useRowFeedback)
                modeName = "PATCH_ROW";
            else if (useColFeedback)
                modeName = "PATCH_COL";

            System.out.println("\n=== MULTI-SEED LEAKAGE-FREE SUMMARY ===");
            System.out.printf("Mode: %s%n", modeName);
            System.out.printf("Seeds: %d  AdaptSize: 2000%n", results.size());
            System.out.printf("FrozenAcc mean=%.4f std=%.4f%n", meanFrozen, stdFrozen);
            System.out.printf("Delta     mean=%+.4f std=%.4f min=%+.4f max=%+.4f%n", meanDelta, stdDelta, minDelta,
                    maxDelta);

            System.out.println("\n=== CSV OUTPUT ===");
            System.out.println("seed,baselineAcc,frozenAcc,delta");
            for (LeakageFreeResult r : results) {
                System.out.printf("%d,%.4f,%.4f,%.4f%n", r.seed, r.baselineAcc, r.frozenAcc, r.getDelta());
            }

            logger.info("============================================================");
            logger.info("MULTI-SEED VERIFICATION COMPLETE");
            logger.info("============================================================");

            if (context != null) {
                context.close();
                System.exit(0);
            }

        } catch (Exception e) {
            logger.error("Multi-seed verification failed", e);
        }
    }

    private void runLeakageFreeVerification(RowColumnDigitClassifier model, List<DigitImage> testData,
            List<DigitImage> trainData, com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model,
            com.markovai.server.ai.DigitGradientUnigramModel gradModel,
            boolean useRowFeedback, boolean useColFeedback) {
        logger.info("============================================================");
        logger.info("STARTING LEAKAGE-FREE VERIFICATION PROTOCOL (Single Seed)");
        logger.info("============================================================");

        try {
            performLeakageFreeProtocol(model, testData, trainData, patch4x4Model, gradModel, 12345L, true,
                    useRowFeedback,
                    useColFeedback, 2000, null);

            logger.info("============================================================");
            logger.info("VERIFICATION PROTOCOL COMPLETE");
            logger.info("============================================================");

            if (context != null) {
                logger.info("Shutting down application...");
                context.close();
                System.exit(0);
            }

        } catch (Exception e) {
            logger.error("Verification failed", e);
        }
    }

    private LeakageFreeResult performLeakageFreeProtocol(RowColumnDigitClassifier model, List<DigitImage> testData,
            List<DigitImage> trainData, com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model,
            com.markovai.server.ai.DigitGradientUnigramModel gradModel,
            long seed, boolean verbose, boolean useRowFeedback, boolean useColFeedback, int adaptSize,
            Double knownBaseline)
            throws Exception {

        // 1. Build MRF
        FactorGraphBuilder builder = new FactorGraphBuilder(
                model.getRowModel(), model.getColumnModel(), model.getPatchModel(),
                model.getRowExtractor(), model.getColumnExtractor(), model.getPatchExtractor(),
                patch4x4Model, gradModel, DataPathResolver.resolveDbPath());

        ObjectMapper mapper = new ObjectMapper();
        FactorGraphBuilder.ConfigRoot configRoot = mapper.readValue(
                getClass().getResourceAsStream("/mrf_config.json"),
                FactorGraphBuilder.ConfigRoot.class);
        Map<String, DigitFactorNode> nodes = builder.build(getClass().getResourceAsStream("/mrf_config.json"));
        DigitFactorNode root = nodes.get(configRoot.rootNodeId);
        if (root == null)
            throw new RuntimeException("Root node not found");

        MarkovFieldDigitClassifier mrf = new MarkovFieldDigitClassifier(root);
        InferenceEngine engine = InferenceEngineFactory.create(configRoot, mrf);

        // Define canonical datasets
        List<DigitImage> phaseATest = testData;
        List<DigitImage> phaseBAdapt = selectAdaptationSubset(trainData, adaptSize, seed);
        List<DigitImage> phaseCTest = testData;

        // Validate Integrity
        validateNoOverlap(phaseBAdapt, phaseCTest);

        if (verbose) {
            logger.info("Leakage-free protocol dataset summary (Seed={}):", seed);
            logger.info("  Train size: {}", trainData.size());
            logger.info("  Test size:  {}", testData.size());
            logger.info("  Adapt size: {}", phaseBAdapt.size());
            logger.info("  Phase A eval size: {}", phaseATest.size());
            logger.info("  Phase C eval size: {}", phaseCTest.size());
        }

        // Load configs from file or defaults
        com.markovai.server.ai.Patch4x4FeedbackConfig basePatchConfig = getFeedbackConfigOrDefault("patch4x4");
        com.markovai.server.ai.Patch4x4FeedbackConfig baseRowConfig = getFeedbackConfigOrDefault("row");
        com.markovai.server.ai.Patch4x4FeedbackConfig baseColConfig = getFeedbackConfigOrDefault("col");

        // 2. Phase A: Baseline (Feedback Disabled)
        if (verbose)
            logger.info("PHASE A: Baseline Test Accuracy (Feedback Disabled)");

        // Apply disabled configs for baseline
        com.markovai.server.ai.Patch4x4FeedbackConfig rowBaseline = baseRowConfig.copy();
        rowBaseline.enabled = false;
        mrf.setRowFeedbackConfig(rowBaseline);

        com.markovai.server.ai.Patch4x4FeedbackConfig colBaseline = baseColConfig.copy();
        colBaseline.enabled = false;
        mrf.setColumnFeedbackConfig(colBaseline);

        com.markovai.server.ai.Patch4x4FeedbackConfig patchBaseline = basePatchConfig.copy();
        patchBaseline.enabled = false;
        mrf.setPatch4x4Config(patchBaseline);

        double baselineAcc;
        if (knownBaseline != null) {
            baselineAcc = knownBaseline;
            if (verbose)
                logger.info("PHASE A: Baseline Test Accuracy (Using Precomputed): {}",
                        String.format("%.4f", baselineAcc));
        } else {
            baselineAcc = mrf.evaluateAccuracy(phaseATest, true, engine);
        }

        // 3. Phase B: Adaptation (Feedback Enabled, Learning Enabled) on TRAIN subset
        if (verbose)
            logger.info("PHASE B: Adaptation on TRAIN subset ({} images). Feedback Learning ENABLED.",
                    phaseBAdapt.size());

        // Setup Adaptation Configs
        com.markovai.server.ai.Patch4x4FeedbackConfig rowAdapt = baseRowConfig.copy();
        if (useRowFeedback) {
            rowAdapt.enabled = true;
            rowAdapt.learningEnabled = true;
        } else {
            rowAdapt.enabled = false;
            rowAdapt.learningEnabled = false;
        }
        mrf.setRowFeedbackConfig(rowAdapt);

        com.markovai.server.ai.Patch4x4FeedbackConfig colAdapt = baseColConfig.copy();
        if (useColFeedback) {
            colAdapt.enabled = true;
            colAdapt.learningEnabled = true;
        } else {
            colAdapt.enabled = false;
            colAdapt.learningEnabled = false;
        }
        mrf.setColumnFeedbackConfig(colAdapt);

        com.markovai.server.ai.Patch4x4FeedbackConfig patchAdapt = basePatchConfig.copy();
        patchAdapt.enabled = true;
        patchAdapt.learningEnabled = true;
        mrf.setPatch4x4Config(patchAdapt);

        // Run evaluation on ADAPT set (isTestSet=false)
        if ("network".equalsIgnoreCase(configRoot.topology)) {
            boolean basinEnabled = (configRoot.learning != null && configRoot.learning.basin != null
                    && Boolean.TRUE.equals(configRoot.learning.basin.enabled));
            boolean payoffEnabled = (configRoot.learning != null && configRoot.learning.payoff != null
                    && Boolean.TRUE.equals(configRoot.learning.payoff.enabled));

            final FactorGraphBuilder.PayoffConfig pCfg = (configRoot.learning != null) ? configRoot.learning.payoff
                    : null;
            final FactorGraphBuilder.BasinConfig bCfg = (configRoot.learning != null) ? configRoot.learning.basin
                    : null;
            final FactorGraphBuilder.ObserverWeightsConfig obsCfg = (configRoot.learning != null)
                    ? configRoot.learning.observerWeights
                    : null;

            final com.markovai.server.ai.learning.ObserverWeightState weightState = (obsCfg != null
                    && Boolean.TRUE.equals(obsCfg.enabled))
                            ? new com.markovai.server.ai.learning.ObserverWeightState(obsCfg)
                            : null;

            if (weightState != null) {
                logger.info("Observer Weight Learning ENABLED. Alpha={}", obsCfg.alpha);
            }
            logger.info(
                    "ObserverWeights enabled={}, standardizeObserverScores={}, WeightedSumNode.standardizeObserverScores={}",
                    (obsCfg != null ? obsCfg.enabled : "null"),
                    (obsCfg != null ? obsCfg.standardizeObserverScores : "null"),
                    mrf.getRootStandardizationStatus());

            if (weightState != null) {
                logger.info("Active Observer IDs for Weight Learning: {}", mrf.getObserverNodeIds());
                // We'll log the specific observers being tracked after we see the first
                // classifier instance
            }

            if (basinEnabled && bCfg != null && payoffEnabled) {
                logger.info("Using BASIN-AWARE Learning: scheme={}, trajectory={}, gate={}", bCfg.scheme,
                        bCfg.useTrajectory, bCfg.minDeltaGate);
                BasinMetrics basinMetrics = new BasinMetrics();

                mrf.evaluateAccuracy(phaseBAdapt, false, engine, (img, res, classifier, trueDigit, scores) -> {
                    // 1. Calculate base Payoff Scale
                    double payoffScale = (pCfg != null && res != null)
                            ? PayoffCalculator.computePayoffScale(res, trueDigit, pCfg)
                            : 1.0;

                    // 2. Identify Winner and Rival from Trajectory
                    int winner = -1;
                    int rival = -1;
                    double winningScore = Double.NEGATIVE_INFINITY;
                    double rivalScore = Double.NEGATIVE_INFINITY;

                    // Using final scores from result if available (which should be b_final if using
                    // NetworkInferenceResult)
                    // or just using the 'scores' passed in.
                    // Note: 'scores' in FeedbackStrategy are typically LogLikelihoods or Softmax
                    // depending on engine.
                    // For Network, getBelief() is better.
                    double[] belief = null;
                    if (res instanceof com.markovai.server.ai.inference.NetworkInferenceResult) {
                        belief = res.getBelief();
                    }

                    if (belief != null) {
                        // Find top2
                        for (int d = 0; d < 10; d++) {
                            if (belief[d] > winningScore) {
                                rivalScore = winningScore;
                                rival = winner;
                                winningScore = belief[d];
                                winner = d;
                            } else if (belief[d] > rivalScore) {
                                rivalScore = belief[d];
                                rival = d;
                            }
                        }
                    } else {
                        // Fallback to scores
                        for (int d = 0; d < 10; d++) {
                            if (scores[d] > winningScore) {
                                rivalScore = winningScore;
                                rival = winner;
                                winningScore = scores[d];
                                winner = d;
                            } else if (scores[d] > rivalScore) {
                                rivalScore = scores[d];
                                rival = d;
                            }
                        }
                    }

                    // 3. Compute Trajectory Weights
                    double sumPosDw = 0.0;
                    double sumPosDr = 0.0;

                    int iters = res.getIterations();
                    boolean useTrajectory = Boolean.TRUE.equals(bCfg.useTrajectory) &&
                            res instanceof com.markovai.server.ai.inference.NetworkInferenceResult;

                    if (useTrajectory && iters >= bCfg.trajectoryMinStep) {
                        com.markovai.server.ai.inference.NetworkInferenceResult netRes = (com.markovai.server.ai.inference.NetworkInferenceResult) res;
                        java.util.List<double[]> traj = netRes.getBeliefTrajectory();

                        if (traj != null && traj.size() > 1) {
                            // t=1 to T
                            for (int t = 1; t < traj.size(); t++) {
                                double[] curr = traj.get(t);
                                double[] prev = traj.get(t - 1);

                                double dw = curr[winner] - prev[winner];
                                double dr = curr[rival] - prev[rival];

                                if (dw > 0)
                                    sumPosDw += dw;
                                if (dr > 0)
                                    sumPosDr += dr;
                            }
                        } else {
                            // Fallback uniform if trajectory missing
                            sumPosDw = 1.0;
                            sumPosDr = 1.0; // Normalized later? No, usually small.
                            // If missing, we treat as uniform?
                            // Step prompt says: "If both sums are ~0, fall back to uniform."
                        }
                    } else {
                        // Fallback uniform
                        sumPosDw = 1.0; // Equivalent to 1.0 multiplier? relative to minDeltaGate?
                        sumPosDr = 0.0; // If fallback, what do we do?
                        // "fall back to existing Step 3 payoffScale behavior (uniform updateScale)"
                        // This means reinforceScale = payoffScale, penalizeScale = payoffScale (if
                        // doing dual update).
                        // Standard update: true=trueDigit, rival=scoreRival.
                    }
                    double wReinforce = 0.0;
                    double wPenalize = 0.0;

                    // 4. Determine Targets and Scales
                    double reinforceScale = 0.0;
                    double penalizeScale = 0.0;
                    int targetW = -1;
                    int targetR = -1;
                    boolean fallback = false;

                    double minGate = (bCfg.minDeltaGate != null) ? bCfg.minDeltaGate : 1e-5;

                    double denom = sumPosDw + sumPosDr + 1e-9;
                    boolean normalize = (bCfg.normalizeTrajectory != null && bCfg.normalizeTrajectory) ||
                            (bCfg.normalizeTrajectory == null && true); // Default true

                    if ((normalize && denom < 1e-8) || (!normalize && sumPosDw < minGate && sumPosDr < minGate)) {
                        // Fallback to uniform standard
                        fallback = true;
                        reinforceScale = payoffScale; // Standard
                        penalizeScale = payoffScale; // Standard
                        // Standard means: True gets +, Rival gets -
                        targetW = trueDigit;
                        targetR = rival; // The rival from belief/scores

                    } else {
                        // Basin Weighted
                        if (winner == trueDigit) {
                            targetW = trueDigit;
                            targetR = rival;
                        } else {
                            targetW = trueDigit;
                            targetR = winner; // The wrong attractor is the rival to penalize
                        }

                        double cap = (bCfg.penalizeCap != null) ? bCfg.penalizeCap : 0.0;
                        double wMult = (bCfg.winnerReinforce != null) ? bCfg.winnerReinforce : 1.0;
                        double rMult = (bCfg.rivalPenalize != null) ? bCfg.rivalPenalize : 0.0;

                        // Enforce Reinforcement-Only if explicitly configured (cap<=0 or rMult<=0)
                        boolean reinforcementOnly = (cap <= 0.0 || rMult <= 0.0);

                        if (normalize) {
                            if (reinforcementOnly) {
                                wPenalize = 0.0;
                                wReinforce = 1.0;
                                reinforceScale = payoffScale * wMult; // Full payoff to winner
                                penalizeScale = 0.0;
                            } else {
                                // Raw weights
                                double wPenalizeRaw = sumPosDr / denom;
                                // Cap penalization
                                wPenalize = Math.min(wPenalizeRaw, cap);

                                // Renormalize reinforce to take the rest
                                wReinforce = 1.0 - wPenalize;

                                reinforceScale = payoffScale * wMult * wReinforce;
                                penalizeScale = payoffScale * rMult * wPenalize;
                            }
                        } else {
                            // Old logic (raw sums) - largely unused now
                            reinforceScale = payoffScale * wMult * sumPosDw;
                            penalizeScale = reinforcementOnly ? 0.0 : (payoffScale * rMult * sumPosDr);
                        }
                    }

                    boolean correctAttractor = (winner == trueDigit);
                    basinMetrics.update(reinforceScale, penalizeScale, fallback, sumPosDw, sumPosDr, correctAttractor,
                            iters, wReinforce, wPenalize, payoffScale);

                    // 5. Apply Updates
                    // Calculate margin for standard logging/gating usage inside node
                    double margin = (belief != null && targetW >= 0 && targetR >= 0)
                            ? (belief[targetW] - belief[targetR])
                            : 0.0;
                    boolean wasCorrect = (res.getPredictedDigit() == trueDigit);

                    if (fallback) {
                        // Standard combined update
                        classifier.applyFeedbackToNodes(img, targetW, targetR, reinforceScale, wasCorrect, margin);
                    } else {
                        // Independent updates
                        // Reinforce Winner (targetW)
                        if (targetW != -1 && reinforceScale > 0) {
                            classifier.applyFeedbackToNodes(img, targetW, -1, reinforceScale, wasCorrect, margin);
                        }
                        // Penalize Rival (targetR)
                        if (targetR != -1 && penalizeScale > 0) {
                            // Penalize means call with: true=-1, rival=targetR, scale=penalizeScale
                            // (since applyFeedback does adj[rival] -= scale)
                            classifier.applyFeedbackToNodes(img, -1, targetR, penalizeScale, wasCorrect, margin);
                        }
                    }

                    // Observer Weight Update (Basin Branch)
                    if (weightState != null && res instanceof com.markovai.server.ai.inference.NetworkInferenceResult) {
                        com.markovai.server.ai.inference.NetworkInferenceResult netRes = (com.markovai.server.ai.inference.NetworkInferenceResult) res;
                        Map<String, double[]> leafScores = netRes.getLeafScores();

                        if (leafScores != null) {
                            java.util.Set<String> obsIds = classifier.getObserverNodeIds();

                            // Convergence Gate
                            boolean checkConv = Boolean.TRUE.equals(obsCfg.requireConvergence);
                            boolean isConverged = (!checkConv) ||
                                    (netRes.getFinalMaxDelta() <= (configRoot.network != null
                                            ? configRoot.network.stopEpsilon
                                            : 1e-4));

                            // Gating
                            boolean skipUpdate = (Boolean.TRUE.equals(obsCfg.updateOnlyIfIncorrect) && wasCorrect)
                                    || !isConverged;

                            if (!skipUpdate) {
                                // Use heuristic: true vs rival
                                int r = (winner == trueDigit) ? rival : winner;

                                for (String nodeId : obsIds) {
                                    double[] ls = leafScores.get(nodeId);
                                    if (ls != null) {
                                        double m = ls[trueDigit] - ls[r];
                                        weightState.update(nodeId, m, payoffScale);
                                    }
                                }
                            }
                            classifier.setObserverWeights(weightState.computeWeights(obsIds));
                        }
                    }

                }); // End of evaluateAccuracy lambda

                basinMetrics.printSummary(logger);
                if (weightState != null) {
                    weightState.logSummary("Phase B (Basin) Final", true);
                }

            } else if (payoffEnabled) {
                FactorGraphBuilder.PayoffConfig pCfgLocal = pCfg;

                mrf.evaluateAccuracy(phaseBAdapt, false, engine, (img, res, classifier, trueDigit, scores) -> {
                    // Payoff Logic
                    double payoffScale = (pCfgLocal != null && res != null)
                            ? PayoffCalculator.computePayoffScale(res, trueDigit, pCfgLocal)
                            : 1.0;

                    // Standard Feedback
                    int rivalDigit = -1;
                    double rivalScore = Double.NEGATIVE_INFINITY;
                    for (int d = 0; d < 10; d++) {
                        if (d == trueDigit)
                            continue;
                        if (scores[d] > rivalScore) {
                            rivalScore = scores[d];
                            rivalDigit = d;
                        }
                    }
                    double margin = scores[trueDigit] - rivalScore;
                    boolean wasCorrect = (res != null) ? (res.getPredictedDigit() == trueDigit) : false;

                    classifier.applyFeedbackToNodes(img, trueDigit, rivalDigit, payoffScale, wasCorrect, margin);

                    // Observer Weight Update
                    if (weightState != null && res instanceof com.markovai.server.ai.inference.NetworkInferenceResult) {
                        com.markovai.server.ai.inference.NetworkInferenceResult netRes = (com.markovai.server.ai.inference.NetworkInferenceResult) res;
                        java.util.Map<String, double[]> leafScores = netRes.getLeafScores();

                        if (leafScores != null) {
                            java.util.Set<String> obsIds = classifier.getObserverNodeIds();

                            // Convergence Gate
                            boolean checkConv = Boolean.TRUE.equals(obsCfg.requireConvergence);
                            boolean isConverged = (!checkConv) ||
                                    (netRes.getFinalMaxDelta() <= (configRoot.network != null
                                            ? configRoot.network.stopEpsilon
                                            : 1e-4));

                            // Gating
                            boolean skipUpdate = (Boolean.TRUE.equals(obsCfg.updateOnlyIfIncorrect) && wasCorrect)
                                    || !isConverged;

                            if (!skipUpdate) {
                                String rule = (obsCfg.updateRule != null) ? obsCfg.updateRule : "heuristic";

                                if ("cross_entropy".equalsIgnoreCase(rule)) {
                                    // 1. Compute Ensemble Distribution p[d]
                                    double temp = (obsCfg.scoreSoftmaxTemperature != null)
                                            ? obsCfg.scoreSoftmaxTemperature
                                            : 1.0;
                                    double[] finalLogScores = netRes.getScores(); // Usually log-likelihoods
                                    double[] probs = new double[10];
                                    double maxS = Double.NEGATIVE_INFINITY;
                                    for (double s : finalLogScores)
                                        if (s > maxS)
                                            maxS = s;
                                    double sumExp = 0.0;
                                    for (int d = 0; d < 10; d++) {
                                        probs[d] = Math.exp((finalLogScores[d] - maxS) / temp);
                                        sumExp += probs[d];
                                    }
                                    for (int d = 0; d < 10; d++)
                                        probs[d] /= sumExp;

                                    // 2. Standardization / Centering
                                    boolean doStandardize = Boolean.TRUE.equals(obsCfg.standardizeObserverScores);
                                    boolean doCenter = Boolean.TRUE.equals(obsCfg.centerObserverScores);

                                    for (java.util.Set<String> activeIds = obsIds; !activeIds.isEmpty();) {
                                        for (String nodeId : activeIds) {
                                            double[] nodeScores = leafScores.get(nodeId);
                                            if (nodeScores == null)
                                                continue;

                                            double[] effectiveScores = nodeScores;

                                            if (doStandardize) {
                                                double mean = 0.0;
                                                for (double v : nodeScores)
                                                    mean += v;
                                                mean /= 10.0;

                                                double var = 0.0;
                                                for (double v : nodeScores)
                                                    var += (v - mean) * (v - mean);
                                                double std = Math.sqrt(var / 10.0) + 1e-6;

                                                double[] z = new double[10];
                                                for (int d = 0; d < 10; d++) {
                                                    z[d] = (nodeScores[d] - mean) / std;
                                                }
                                                effectiveScores = z;
                                            } else if (doCenter && "per_image_mean".equals(obsCfg.centerMode)) {
                                                double mean = 0.0;
                                                for (double v : nodeScores)
                                                    mean += v;
                                                mean /= 10.0;
                                                double[] centered = new double[10];
                                                for (int d = 0; d < 10; d++)
                                                    centered[d] = nodeScores[d] - mean;
                                                effectiveScores = centered;
                                            }

                                            weightState.updateCrossEntropy(nodeId, effectiveScores, trueDigit, probs,
                                                    payoffScale);
                                        }
                                        break; // effectively just iterator
                                    }
                                } else {
                                    // Legacy Heuristic
                                    int r = (wasCorrect) ? PayoffCalculator.findRival(netRes.getBelief(), trueDigit)
                                            : netRes.getPredictedDigit();

                                    for (String nodeId : obsIds) {
                                        double[] ls = leafScores.get(nodeId);
                                        if (ls != null) {
                                            double m = ls[trueDigit] - ls[r];
                                            weightState.update(nodeId, m, payoffScale);
                                        }
                                    }
                                }
                            }
                            classifier.setObserverWeights(weightState.computeWeights(obsIds));
                            classifier.setObserverWeights(weightState.computeWeights(obsIds));
                        }
                    } else {
                        // Ensure no weights are active if learning is disabled
                        classifier.setObserverWeights(null);
                    }
                });

                if (weightState != null) {
                    weightState.logSummary("Phase B (Payoff) Final", true);
                }

            } else {
                mrf.evaluateAccuracy(phaseBAdapt, false, engine);
            }
        } else {
            // Not network topology
            mrf.evaluateAccuracy(phaseBAdapt, false, engine);
        }

        if (verbose)
            logger.info("Adaptation complete.");

        // 4. Phase C: Final Test (Feedback Enabled, Learning FROZEN)
        if (verbose)
            logger.info("PHASE C: Final Test Accuracy (Feedback Scoring Enabled, Learning Frozen)");

        // Freeze Row Feedback
        com.markovai.server.ai.Patch4x4FeedbackConfig rowFrozen = rowAdapt
                .copy();
        rowFrozen.learningEnabled = false;
        mrf.setRowFeedbackConfig(rowFrozen);

        // Freeze Col Feedback
        com.markovai.server.ai.Patch4x4FeedbackConfig colFrozen = colAdapt
                .copy();
        colFrozen.learningEnabled = false;
        mrf.setColumnFeedbackConfig(colFrozen);

        // Freeze Patch Feedback
        com.markovai.server.ai.Patch4x4FeedbackConfig patchFrozen = patchAdapt
                .copy();
        patchFrozen.learningEnabled = false;
        mrf.setPatch4x4Config(patchFrozen);

        // Run on Test Data (isTestSet=true)
        double frozenAcc = mrf.evaluateAccuracy(phaseCTest, true, engine);

        return new LeakageFreeResult(seed, baselineAcc, frozenAcc);
    }

    private void runAdaptationSizeSweep(RowColumnDigitClassifier model, List<DigitImage> trainData,
            List<DigitImage> testData, com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model,
            com.markovai.server.ai.DigitGradientUnigramModel gradModel) {
        logger.info("============================================================");
        logger.info("STARTING ADAPTATION SIZE SWEEP (LEAKAGE-FREE)");
        logger.info("============================================================");

        try {
            // Filter training data to ensure no overlap with test data
            java.util.Set<String> testHashes = testData.stream()
                    .map(img -> img.imageHash)
                    .filter(java.util.Objects::nonNull)
                    .collect(java.util.stream.Collectors.toSet());

            List<DigitImage> cleanTrainData = trainData.stream()
                    .filter(img -> img.imageHash == null || !testHashes.contains(img.imageHash))
                    .collect(java.util.stream.Collectors.toList());

            if (cleanTrainData.size() < trainData.size()) {
                logger.warn("Removed {} overlapping images from training set to ensure leakage-free protocol.",
                        trainData.size() - cleanTrainData.size());
            }

            String adaptSizesStr = System.getProperty("adaptSizes");
            int[] adaptSizes;
            if (adaptSizesStr != null && !adaptSizesStr.isEmpty()) {
                String[] parts = adaptSizesStr.split(",");
                adaptSizes = new int[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    adaptSizes[i] = Integer.parseInt(parts[i].trim());
                }
            } else {
                adaptSizes = new int[] { 2000, 5000, 10000, 20000 };
            }

            String seedsStr = System.getProperty("adaptSeeds");
            long[] seeds;
            if (seedsStr != null && !seedsStr.isEmpty()) {
                String[] parts = seedsStr.split(",");
                seeds = new long[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    seeds[i] = Long.parseLong(parts[i].trim());
                }
            } else {
                seeds = new long[] { 12345L, 22222L, 33333L, 44444L, 55555L };
            }

            List<String> modes = new ArrayList<>();
            String modeParam = System.getProperty("feedbackMode");
            String modesParam = System.getProperty("feedbackModes");

            if ("ALL".equalsIgnoreCase(modeParam)) {
                modes.add("PATCH");
                modes.add("PATCH_ROW");
                modes.add("PATCH_COL");
                modes.add("PATCH_ROW_COL");
            } else if (modesParam != null && !modesParam.isEmpty()) {
                for (String m : modesParam.split(",")) {
                    modes.add(m.trim().toUpperCase());
                }
            } else if (modeParam != null && !modeParam.isEmpty()) {
                modes.add(modeParam.toUpperCase());
            } else {
                modes.add("PATCH");
                modes.add("PATCH_ROW");
                modes.add("PATCH_COL");
                modes.add("PATCH_ROW_COL");
            }

            double baselineAcc = evaluateSweepBaseline(model, testData, patch4x4Model, gradModel);

            System.out.println("\n=== LEAKAGE-FREE ADAPTATION SIZE SWEEP ===");
            System.out.printf("baselineAcc=%.4f%n", baselineAcc);
            System.out.print("seeds=[");
            for (int i = 0; i < seeds.length; i++)
                System.out.print(seeds[i] + (i < seeds.length - 1 ? "," : ""));
            System.out.println("]");
            System.out.print("adaptSizes=[");
            for (int i = 0; i < adaptSizes.length; i++)
                System.out.print(adaptSizes[i] + (i < adaptSizes.length - 1 ? "," : ""));
            System.out.println("]");

            com.markovai.server.ai.Patch4x4FeedbackConfig cfg = getBaseConfigFromFile();
            System.out.printf(
                    "config: freqScaling=%s, freqScalingMode=%s, applyDecayEveryNUpdates=%d, eta=%.4f, adjScale=%.4f%n",
                    cfg.frequencyScalingEnabled, cfg.frequencyScalingMode, cfg.applyDecayEveryNUpdates, cfg.eta,
                    cfg.adjScale);

            System.out.println("\nmode,adaptSize,meanFrozen,meanDelta,stdDelta,minDelta,maxDelta");

            List<String> verifyRows = new ArrayList<>();

            for (String mode : modes) {
                boolean useRow = mode.contains("ROW");
                boolean useCol = mode.contains("COL");

                for (int adaptSize : adaptSizes) {
                    if (adaptSize > cleanTrainData.size()) {
                        logger.warn("Adaptation size {} exceeds train data size {}, clamping.", adaptSize,
                                cleanTrainData.size());
                    }

                    List<Double> deltas = new ArrayList<>();
                    List<Double> frozens = new ArrayList<>();

                    List<LeakageFreeResult> seedResults = java.util.stream.LongStream.of(seeds).parallel()
                            .mapToObj(seed -> {
                                try {
                                    return performLeakageFreeProtocol(model, testData, cleanTrainData, patch4x4Model,
                                            gradModel,
                                            seed, false, useRow, useCol, adaptSize, baselineAcc);
                                } catch (Exception e) {
                                    throw new RuntimeException(e);
                                }
                            }).collect(java.util.stream.Collectors.toList());

                    for (LeakageFreeResult res : seedResults) {
                        double d = res.frozenAcc - baselineAcc;
                        deltas.add(d);
                        frozens.add(res.frozenAcc);
                    }

                    double sumFrozen = 0;
                    for (double f : frozens)
                        sumFrozen += f;
                    double meanFrozen = sumFrozen / frozens.size();

                    double sumDelta = 0;
                    double minDelta = Double.MAX_VALUE;
                    double maxDelta = -Double.MAX_VALUE;
                    for (double d : deltas) {
                        sumDelta += d;
                        if (d < minDelta)
                            minDelta = d;
                        if (d > maxDelta)
                            maxDelta = d;
                    }
                    double meanDelta = sumDelta / deltas.size();

                    double varDelta = 0;
                    if (deltas.size() > 1) {
                        for (double d : deltas)
                            varDelta += Math.pow(d - meanDelta, 2);
                        varDelta /= (deltas.size() - 1);
                    }
                    double stdDelta = Math.sqrt(varDelta);

                    String rowStr = String.format("%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f",
                            mode, adaptSize, meanFrozen, meanDelta, stdDelta, minDelta, maxDelta);
                    System.out.println(rowStr);
                    verifyRows.add(rowStr);
                }
            }

            System.out.println("\n```csv");
            System.out.println("mode,adaptSize,meanFrozen,meanDelta,stdDelta,minDelta,maxDelta");
            for (String r : verifyRows)
                System.out.println(r);
            System.out.println("```");

            if (context != null) {
                context.close();
                System.exit(0);
            }

        } catch (Exception e) {
            logger.error("Adaptation sweep failed", e);
        }
    }

    private void runFeedbackSweep(RowColumnDigitClassifier model, List<DigitImage> trainData, List<DigitImage> testData,
            com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model,
            com.markovai.server.ai.DigitGradientUnigramModel gradModel) {
        logger.info("============================================================");
        logger.info("STARTING FEEDBACK HYPERPARAMETER SWEEP");
        logger.info("============================================================");

        try {
            // 1. Compute Baseline (Once)
            logger.info("Computing Baseline Accuracy...");
            double baselineAcc = evaluateSweepBaseline(model, testData, patch4x4Model, gradModel);
            logger.info("Baseline Accuracy: {}", String.format("%.4f", baselineAcc));

            // Explicit Grid
            double[] adjScales = { 0.10, 0.05, 0.02, 0.01 };
            double[] etas = { 0.003, 0.001, 0.0005 };

            List<SweepResult> results = new ArrayList<>();

            for (double scale : adjScales) {
                for (double eta : etas) {
                    logger.info("Running sweep iteration: adjScale={}, eta={}", scale, eta);
                    // MRF is rebuilt every iteration, effectively resetting node state.

                    double adaptedAcc = runSweepIteration(model, trainData, testData, patch4x4Model, gradModel, scale,
                            eta);

                    results.add(new SweepResult(scale, eta, baselineAcc, adaptedAcc));
                }
            }

            // Print Full Table
            System.out.println("\n=== FULL LEAKAGE-FREE FEEDBACK SWEEP RESULTS ===");
            System.out.println("freqScalingEnabled=true");
            System.out.println("freqScalingMode=GLOBAL_SQRT");
            System.out.println("decayEveryNUpdates=5000");
            System.out.println("adaptationSetSize=2000");
            System.out.println();
            System.out.println("adjScale\teta\tbaselineAcc\tadaptedAcc\tdelta");
            System.out.println("-------------------------------------------------------");

            for (SweepResult r : results) {
                System.out.printf("%.4f\t%.4f\t%.4f\t%.4f\t%.4f%n",
                        r.adjScale, r.eta, r.baselineAcc, r.adaptedAcc, r.getDelta());
            }

            // CSV Output
            boolean printCsv = "true".equalsIgnoreCase(System.getProperty("printSweepAsCSV")) ||
                    (appArgs != null && appArgs.containsOption("printSweepAsCSV")
                            && "true".equalsIgnoreCase(appArgs.getOptionValues("printSweepAsCSV").get(0)));

            if (printCsv) {
                System.out.println("\n=== CSV OUTPUT ===");
                System.out.println("adjScale,eta,baselineAcc,adaptedAcc,delta");
                for (SweepResult r : results) {
                    System.out.printf("%.4f,%.4f,%.4f,%.4f,%.4f%n",
                            r.adjScale, r.eta, r.baselineAcc, r.adaptedAcc, r.getDelta());
                }
            }

            logger.info("============================================================");
            logger.info("SWEEP COMPLETE");
            logger.info("============================================================");

            if (context != null) {
                context.close();
                System.exit(0);
            }

        } catch (Exception e) {
            logger.error("Sweep failed", e);
        }
    }

    private static class SweepResult {
        double adjScale;
        double eta;
        double baselineAcc;
        double adaptedAcc;

        public SweepResult(double adjScale, double eta, double baselineAcc, double adaptedAcc) {
            this.adjScale = adjScale;
            this.eta = eta;
            this.baselineAcc = baselineAcc;
            this.adaptedAcc = adaptedAcc;
        }

        public double getDelta() {
            return adaptedAcc - baselineAcc;
        }
    }

    private double evaluateSweepBaseline(RowColumnDigitClassifier model, List<DigitImage> testData,
            com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model,
            com.markovai.server.ai.DigitGradientUnigramModel gradModel) throws Exception {
        FactorGraphBuilder.ConfigRoot config = loadMrfConfig();
        MarkovFieldDigitClassifier mrf = buildMrf(model, patch4x4Model, gradModel, config);
        InferenceEngine engine = InferenceEngineFactory.create(config, mrf);
        mrf.setPatch4x4Config(com.markovai.server.ai.Patch4x4FeedbackConfig.disabled());
        return mrf.evaluateAccuracy(testData, true, engine);
    }

    private double runSweepIteration(RowColumnDigitClassifier model, List<DigitImage> trainData,
            List<DigitImage> testData, com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model,
            com.markovai.server.ai.DigitGradientUnigramModel gradModel, double adjScale,
            double eta) throws Exception {

        // Build Fresh MRF
        FactorGraphBuilder.ConfigRoot configRoot = loadMrfConfig();
        MarkovFieldDigitClassifier mrf = buildMrf(model, patch4x4Model, gradModel, configRoot);
        InferenceEngine engine = InferenceEngineFactory.create(configRoot, mrf);

        // Get Base Config (so we respect file settings for other params)
        com.markovai.server.ai.Patch4x4FeedbackConfig baseConfig = getBaseConfigFromFile();

        // Adaptation Train Subset (2000) - Deterministic
        List<DigitImage> adaptSet = selectAdaptationSubset(trainData, 2000, 12345L);

        // Enable Feedback (Adaptation Mode)
        com.markovai.server.ai.Patch4x4FeedbackConfig adaptationCfg = baseConfig.copy();
        adaptationCfg.enabled = true;
        adaptationCfg.learningEnabled = true;
        adaptationCfg.adjScale = adjScale; // Override from sweep
        adaptationCfg.eta = eta; // Override from sweep

        mrf.setPatch4x4Config(adaptationCfg);
        mrf.evaluateAccuracy(adaptSet, false, engine);

        // Freeze and Test
        com.markovai.server.ai.Patch4x4FeedbackConfig finalCfg = adaptationCfg.copy();
        finalCfg.learningEnabled = false;
        // Keep adjScale / eta same for inference (scoring)

        mrf.setPatch4x4Config(finalCfg);

        return mrf.evaluateAccuracy(testData, true, engine);
    }

    private MarkovFieldDigitClassifier buildMrf(RowColumnDigitClassifier model,
            com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model,
            com.markovai.server.ai.DigitGradientUnigramModel gradModel, FactorGraphBuilder.ConfigRoot configRoot)
            throws Exception {
        FactorGraphBuilder builder = new FactorGraphBuilder(
                model.getRowModel(), model.getColumnModel(), model.getPatchModel(),
                model.getRowExtractor(), model.getColumnExtractor(), model.getPatchExtractor(),
                patch4x4Model, gradModel, DataPathResolver.resolveDbPath());

        Map<String, DigitFactorNode> nodes = builder.build(getClass().getResourceAsStream("/mrf_config.json"));
        DigitFactorNode root = nodes.get(configRoot.rootNodeId);
        if (root == null)
            throw new RuntimeException("Root node not found");

        return new MarkovFieldDigitClassifier(root);
    }

    private FactorGraphBuilder.ConfigRoot loadMrfConfig() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(
                getClass().getResourceAsStream("/mrf_config.json"),
                FactorGraphBuilder.ConfigRoot.class);
    }

    /**
     * Finds the base configuration from the loaded file.
     * Deprecated in favor of getFeedbackConfigOrDefault("patch4x4").
     */
    private com.markovai.server.ai.Patch4x4FeedbackConfig getBaseConfigFromFile() throws Exception {
        return getFeedbackConfigOrDefault("patch4x4");
    }

    /**
     * Loads feedback config for a node by ID.
     * If config is missing in JSON, falls back to hardcoded defaults:
     * - "patch4x4": existing default
     * - "row" / "col": existing row/col default (with enabled=false, which is
     * overridden by protocol)
     */
    com.markovai.server.ai.Patch4x4FeedbackConfig getFeedbackConfigOrDefault(String nodeId) {
        try {
            FactorGraphBuilder.ConfigRoot root = loadMrfConfig();
            for (FactorGraphBuilder.ConfigNode node : root.nodes) {
                if (nodeId.equals(node.id) && node.feedback != null) {
                    return node.feedback;
                }
            }
        } catch (Exception e) {
            logger.warn("Failed to load feedback config for node {}, using defaults. Error: {}", nodeId,
                    e.getMessage());
        }

        // Fallback Defaults
        if ("patch4x4".equals(nodeId)) {
            return new com.markovai.server.ai.Patch4x4FeedbackConfig(
                    false, false, 0.10, 0.003, 0.02, true, true, 5.0, false, 1.0e-4,
                    false, "GLOBAL_SQRT", 0.05, 1.0, 0);
        } else {
            // Default for Row/Col matches previous hardcoded values
            com.markovai.server.ai.Patch4x4FeedbackConfig c = com.markovai.server.ai.Patch4x4FeedbackConfig
                    .disabled();
            c.adjScale = 0.10;
            c.eta = 0.003;
            c.marginTarget = 0.02;
            c.updateOnlyIfIncorrect = true;
            c.useMarginGating = true;
            c.maxAdjAbs = 5.0;
            c.frequencyScalingEnabled = true;
            c.frequencyScalingMode = "GLOBAL_SQRT";
            // Note: applyDecayEveryNUpdates defaults to 0 in disabled(), whereas config
            // might set it to 5000.
            // This maintains backward compatibility if config is missing.
            return c;
        }
    }

    private List<DigitImage> selectAdaptationSubset(List<DigitImage> trainData, int adaptSize, long seed) {
        List<DigitImage> copy = new ArrayList<>(trainData);
        java.util.Collections.shuffle(copy, new java.util.Random(seed));
        return copy.subList(0, Math.min(adaptSize, copy.size()));
    }

    private void validateNoOverlap(List<DigitImage> adaptSet, List<DigitImage> testSet) {
        java.util.Set<String> adaptHashes = new java.util.HashSet<>();
        for (DigitImage img : adaptSet) {
            if (img.imageHash != null)
                adaptHashes.add(img.imageHash);
        }

        int overlapCount = 0;
        for (DigitImage img : testSet) {
            if (img.imageHash != null && adaptHashes.contains(img.imageHash)) {
                overlapCount++;
            }
        }

        if (overlapCount > 0) {
            logger.error("DATA LEAKAGE DETECTED: {} images in adaptation set are also in the test set!", overlapCount);
            throw new IllegalStateException("Data leakage detected: adaptation set overlaps with test set.");
        }
        logger.info("Data integrity check passed: 0 overlaps between adaptation set and test set.");
    }

    private void runNetworkAttractorSanityCheck(RowColumnDigitClassifier model, List<DigitImage> testData,
            com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model,
            com.markovai.server.ai.DigitGradientUnigramModel gradModel) {
        logger.info("============================================================");
        logger.info("STARTING NETWORK ATTRACTOR SANITY CHECK");
        logger.info("============================================================");

        try {
            // 1. Build MRF with Network Config Enforced
            FactorGraphBuilder.ConfigRoot configRoot = loadMrfConfig();

            // Override specific network settings for the test if needed, or rely on file
            // Ensuring network mode is used even if config says layered
            configRoot.topology = "network";
            if (configRoot.network == null) {
                configRoot.network = new FactorGraphBuilder.NetworkConfig();
                configRoot.network.enabled = true;
                configRoot.network.maxIters = 10;
                configRoot.network.debugStats = true; // Enable debug stats
                logger.info("Injecting default network config for sanity check.");
            } else {
                // Ensure enabled and debug
                configRoot.network.enabled = true;
                configRoot.network.debugStats = true;
            }

            MarkovFieldDigitClassifier mrf = buildMrf(model, patch4x4Model, gradModel, configRoot);

            // Force disable all feedback so we test PURE attractor inference on static
            // graph
            mrf.setRowFeedbackConfig(com.markovai.server.ai.Patch4x4FeedbackConfig.disabled());
            mrf.setColumnFeedbackConfig(com.markovai.server.ai.Patch4x4FeedbackConfig.disabled());
            mrf.setPatch4x4Config(com.markovai.server.ai.Patch4x4FeedbackConfig.disabled());

            InferenceEngine engine = InferenceEngineFactory.create(configRoot, mrf);

            // 2. Select Subset
            List<DigitImage> subset = testData.subList(0, Math.min(200, testData.size()));
            logger.info("Running inference on {} images...", subset.size());

            int convergedCount = 0;
            int totalIters = 0;
            double entropyReductionSum = 0.0;
            int correct = 0;

            for (DigitImage img : subset) {
                com.markovai.server.ai.inference.InferenceResult res = engine.infer(img);

                if (res.getIterations() < configRoot.network.maxIters) {
                    convergedCount++;
                }
                totalIters += res.getIterations();

                // Estimate entropy reduction: H(initial) - H(final)
                // We don't have H(initial) explicitly returned in result, but we can recompute
                // if needed.
                // Or just trust the logs.
                // Let's rely on iteration count and convergence as main metrics.
                // Actually, let's compute initial entropy from base scores if we want precise
                // number,
                // but for sanity check, just iteration count is good signal.

                if (res.getPredictedDigit() == img.label) {
                    correct++;
                }
            }

            double meanIters = (double) totalIters / subset.size();
            double accuracy = (double) correct / subset.size();

            logger.info("Sanity Results:");
            logger.info("  Images: {}", subset.size());
            logger.info("  Converged (< maxIters): {}/{}", convergedCount, subset.size());
            logger.info("  Mean Iterations: {:.2f}", meanIters);
            logger.info("  Accuracy (No Learning): {:.4f}", accuracy);

            if (meanIters > 1.0) {
                logger.info("SUCCESS: Iterative refinement is active (mean iters > 1.0)");
            } else {
                logger.warn("WARNING: Mean iterations is 1.0. Attractor loop might be inactive or trivial.");
            }

            logger.info("============================================================");
            logger.info("NETWORK SANITY CHECK COMPLETE");
            logger.info("============================================================");

            if (context != null) {
                context.close();
                System.exit(0);
            }

        } catch (Exception e) {
            logger.error("Network sanity check failed", e);
        }
    }

    private void runNetworkConvergenceSweep(RowColumnDigitClassifier model, List<DigitImage> testData,
            com.markovai.server.ai.DigitPatch4x4UnigramModel patch4x4Model,
            com.markovai.server.ai.DigitGradientUnigramModel gradModel) {
        logger.info("============================================================");
        logger.info("STARTING NETWORK CONVERGENCE TUNING SWEEP");
        logger.info("============================================================");

        try {
            // 1. Setup deterministic subset of 2000 images
            List<DigitImage> subset = new ArrayList<>(testData);
            java.util.Collections.shuffle(subset, new java.util.Random(12345L));
            if (subset.size() > 2000) {
                subset = subset.subList(0, 2000);
            }
            logger.info("Using subset of {} images for sweep.", subset.size());

            // 2. Define Parameter Grid
            double[] priorWeights = { 0.05, 0.10, 0.20, 0.30 };
            double[] dampings = { 0.2, 0.4, 0.6, 0.8 };
            int[] maxItersList = { 10, 20 };
            double[] stopEpsilons = { 1e-4, 5e-4, 1e-3 };

            // 3. Prepare Config
            FactorGraphBuilder.ConfigRoot baseConfig = loadMrfConfig();
            baseConfig.topology = "network";
            if (baseConfig.network == null) {
                baseConfig.network = new FactorGraphBuilder.NetworkConfig();
                baseConfig.network.enabled = true;
            }
            // Fix constants
            baseConfig.network.temperature = 1.0;
            baseConfig.network.epsilon = 1e-9;
            baseConfig.network.debugStats = false; // Disable debug logs to reduce noise

            // Disable feedback learning to ensure stability
            MarkovFieldDigitClassifier mrf = buildMrf(model, patch4x4Model, gradModel, baseConfig);
            mrf.setRowFeedbackConfig(com.markovai.server.ai.Patch4x4FeedbackConfig.disabled());
            mrf.setColumnFeedbackConfig(com.markovai.server.ai.Patch4x4FeedbackConfig.disabled());
            mrf.setPatch4x4Config(com.markovai.server.ai.Patch4x4FeedbackConfig.disabled());

            // We need to rebuild MRF or engine?
            // The MRF structure is constant. The InferenceEngine takes the config.
            // We can reuse the ONE mrf instance, but we need to create a NEW
            // InferenceEngine for each config param set.

            List<String> results = new ArrayList<>();
            results.add(
                    "priorWeight,damping,maxIters,stopEpsilon,convergeRate,meanIters,medianIters,p90Iters,meanFinalMaxDelta,meanEntropyInit,meanEntropyFinal,meanEntropyRed,oscillationCount,accuracy");

            logger.info("Total combinations: {}",
                    priorWeights.length * dampings.length * maxItersList.length * stopEpsilons.length);

            for (double prior : priorWeights) {
                for (double damp : dampings) {
                    for (int maxIter : maxItersList) {
                        for (double stopEps : stopEpsilons) {
                            // Update Config
                            baseConfig.network.priorWeight = prior;
                            baseConfig.network.damping = damp;
                            baseConfig.network.maxIters = maxIter;
                            baseConfig.network.stopEpsilon = stopEps;

                            // Create Engine
                            InferenceEngine engine = InferenceEngineFactory.create(baseConfig, mrf);

                            // Run Inference
                            int converged = 0;
                            long sumIters = 0;
                            List<Integer> itersList = new ArrayList<>();
                            double sumFinalDelta = 0;
                            double sumEntInit = 0;
                            double sumEntFinal = 0;
                            int oscillationCount = 0;
                            int correct = 0;

                            for (DigitImage img : subset) {
                                com.markovai.server.ai.inference.NetworkInferenceResult res = (com.markovai.server.ai.inference.NetworkInferenceResult) engine
                                        .infer(img);

                                int it = res.getIterations();
                                if (it < maxIter) {
                                    converged++;
                                }
                                sumIters += it;
                                itersList.add(it);

                                sumFinalDelta += res.getFinalMaxDelta();
                                sumEntInit += res.getInitialEntropy();
                                sumEntFinal += res.getFinalEntropy();
                                if (res.isOscillationDetected()) {
                                    oscillationCount++;
                                }
                                if (res.getPredictedDigit() == img.label) {
                                    correct++;
                                }
                            }

                            // Compute Stats
                            double convergeRate = (double) converged / subset.size();
                            double meanIters = (double) sumIters / subset.size();
                            java.util.Collections.sort(itersList);
                            double medianIters = itersList.get(subset.size() / 2);
                            double p90Iters = itersList.get((int) (subset.size() * 0.9));
                            double meanFinalDelta = sumFinalDelta / subset.size();
                            double meanEntInit = sumEntInit / subset.size();
                            double meanEntFinal = sumEntFinal / subset.size();
                            double meanEntRed = meanEntInit - meanEntFinal;
                            double accuracy = (double) correct / subset.size();

                            String line = String.format(java.util.Locale.US,
                                    "%.2f,%.2f,%d,%.1e,%.4f,%.2f,%.1f,%.1f,%.6f,%.4f,%.4f,%.4f,%d,%.4f",
                                    prior, damp, maxIter, stopEps, convergeRate, meanIters, medianIters, p90Iters,
                                    meanFinalDelta, meanEntInit, meanEntFinal, meanEntRed, oscillationCount, accuracy);

                            results.add(line);
                            System.out.println(line); // Stream to stdout
                        }
                    }
                }
            }

            // Print Full CSV Block
            System.out.println("\n=== SWEEP RESULTS CSV ===");
            for (String line : results) {
                System.out.println(line);
            }
            System.out.println("=========================\n");

            if (context != null) {
                context.close();
                System.exit(0);
            }

        } catch (Exception e) {
            logger.error("Network sweep failed", e);
        }
    }
}
