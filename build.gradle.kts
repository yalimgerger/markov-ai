plugins {
    id("org.springframework.boot") version "3.2.1" apply false
    id("io.spring.dependency-management") version "1.1.4" apply false
}

allprojects {
    group = "com.markovai"
    version = "0.0.1-SNAPSHOT"

    repositories {
        mavenCentral()
    }
}
