import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.4.10"
}
group = "me.mariakhalusova"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    jcenter()
    maven(url = "https://kotlin.bintray.com/kotlin-datascience")
}

dependencies {
    implementation ("org.jetbrains.kotlin-deeplearning:api:0.0.14")
}
tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}