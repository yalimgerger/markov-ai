package com.markovai.db;

import java.sql.*;
import java.util.Optional;

public class DigitImageDao {

    private final String dbPath;

    public DigitImageDao(String dbPath) {
        this.dbPath = dbPath;
    }

    private Connection connect() throws SQLException {
        return DriverManager.getConnection("jdbc:sqlite:" + dbPath);
    }

    public DigitImage getOrCreateByPath(String imageRelPath, String imageHashOrNull) throws SQLException {
        try (Connection conn = connect()) {
            // Optimistic find first
            Optional<DigitImage> existing = findByPathInternal(conn, imageRelPath);
            if (existing.isPresent()) {
                DigitImage img = existing.get();
                // Update hash if changed and provided
                if (imageHashOrNull != null && !imageHashOrNull.equals(img.getImageHash())) {
                    try (PreparedStatement ps = conn.prepareStatement(
                            "UPDATE digit_image SET image_hash = ? WHERE id = ?")) {
                        ps.setString(1, imageHashOrNull);
                        ps.setLong(2, img.getId());
                        ps.executeUpdate();
                    }
                    return new DigitImage(img.getId(), img.getImageRelPath(), imageHashOrNull, img.getCreatedTs());
                }
                return img;
            }

            // Insert
            long now = System.currentTimeMillis();
            try (PreparedStatement ps = conn.prepareStatement(
                    "INSERT INTO digit_image (image_rel_path, image_hash, created_ts) VALUES (?, ?, ?)",
                    Statement.RETURN_GENERATED_KEYS)) {
                ps.setString(1, imageRelPath);
                ps.setString(2, imageHashOrNull);
                ps.setLong(3, now);
                ps.executeUpdate();

                try (ResultSet rs = ps.getGeneratedKeys()) {
                    if (rs.next()) {
                        long id = rs.getLong(1);
                        return new DigitImage(id, imageRelPath, imageHashOrNull, now);
                    } else {
                        throw new SQLException("Creating digit_image failed, no ID obtained.");
                    }
                }
            } catch (SQLException e) {
                // Handle race condition where it was created in between
                if (e.getMessage().contains("UNIQUE constraint failed")) {
                    return findByPathInternal(conn, imageRelPath)
                            .orElseThrow(() -> new SQLException(
                                    "Failed to find image after UNIQUE constraint violation", e));
                }
                throw e;
            }
        }
    }

    public Optional<DigitImage> findByPath(String imageRelPath) throws SQLException {
        try (Connection conn = connect()) {
            return findByPathInternal(conn, imageRelPath);
        }
    }

    private Optional<DigitImage> findByPathInternal(Connection conn, String imageRelPath) throws SQLException {
        String sql = "SELECT id, image_rel_path, image_hash, created_ts FROM digit_image WHERE image_rel_path = ?";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, imageRelPath);
            try (ResultSet rs = ps.executeQuery()) {
                if (rs.next()) {
                    return Optional.of(new DigitImage(
                            rs.getLong("id"),
                            rs.getString("image_rel_path"),
                            rs.getString("image_hash"),
                            rs.getLong("created_ts")));
                }
            }
        }
        return Optional.empty();
    }
}
