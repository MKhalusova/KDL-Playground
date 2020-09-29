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
    implementation("com.beust:klaxon:5.0.1")
    implementation ("org.jetbrains.kotlin.kotlin-dl:api:0.0.5")
    testImplementation(kotlin("test-junit"))
    implementation(kotlin("script-runtime"))
}
tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}