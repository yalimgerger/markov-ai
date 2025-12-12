package com.markovai.server.ai;

import com.markovai.db.DigitImage;
import com.markovai.db.DigitImageDao;
import com.markovai.db.MarkovChainResultDao;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.SQLException;
import java.util.Optional;

public class CachedMarkovChainEvaluator {

    private static final Logger logger = LoggerFactory.getLogger(CachedMarkovChainEvaluator.class);

    private final MarkovChainEvaluator delegate;
    private final DigitImageDao imageDao;
    private final MarkovChainResultDao resultDao;

    public CachedMarkovChainEvaluator(MarkovChainEvaluator delegate,
            DigitImageDao imageDao,
            MarkovChainResultDao resultDao) {
        this.delegate = delegate;
        this.imageDao = imageDao;
        this.resultDao = resultDao;
    }

    public double[] evaluate(String imageRelPath, String imageHashOrNull, byte[] binary28x28) {
        try {
            // 1. Get or Create Image
            DigitImage img = imageDao.getOrCreateByPath(imageRelPath, imageHashOrNull);

            // 2. Try to load from cache
            String type = delegate.getChainType();
            String version = delegate.getChainVersion();
            Optional<double[]> cached = resultDao.loadScores(img.getId(), type, version);

            if (cached.isPresent()) {
                logger.debug("Cache HIT for image {} chain {}/{}", imageRelPath, type, version);
                return cached.get();
            }

            // 3. Compute
            logger.debug("Cache MISS for image {} chain {}/{}", imageRelPath, type, version);
            double[] scores = delegate.computeScores(binary28x28);

            // 4. Store
            resultDao.upsertScores(img.getId(), type, version, scores);

            return scores;

        } catch (SQLException e) {
            logger.error("Database error in CachedMarkovChainEvaluator, falling back to direct computation", e);
            return delegate.computeScores(binary28x28);
        }
    }

    public String getChainType() {
        return delegate.getChainType();
    }
}
