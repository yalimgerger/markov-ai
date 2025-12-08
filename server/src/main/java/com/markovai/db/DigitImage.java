package com.markovai.db;

public class DigitImage {
    private final long id;
    private final String imageRelPath;
    private final String imageHash;
    private final long createdTs;

    public DigitImage(long id, String imageRelPath, String imageHash, long createdTs) {
        this.id = id;
        this.imageRelPath = imageRelPath;
        this.imageHash = imageHash;
        this.createdTs = createdTs;
    }

    public long getId() {
        return id;
    }

    public String getImageRelPath() {
        return imageRelPath;
    }

    public String getImageHash() {
        return imageHash;
    }

    public long getCreatedTs() {
        return createdTs;
    }

    @Override
    public String toString() {
        return "DigitImage{id=" + id + ", path='" + imageRelPath + "'}";
    }
}
