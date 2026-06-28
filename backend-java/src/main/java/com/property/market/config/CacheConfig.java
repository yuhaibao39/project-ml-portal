package com.property.market.config;

import com.github.benmanes.caffeine.cache.Caffeine;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.cache.CacheManager;
import org.springframework.cache.caffeine.CaffeineCacheManager;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.TimeUnit;

@Configuration
public class CacheConfig {

    @Bean
    public CacheManager cacheManager() {
        CaffeineCacheManager cacheManager = new CaffeineCacheManager(
                "predictions", "modelInfo", "marketStatistics", "trends", "regionalBreakdown"
        );
        cacheManager.setCaffeine(Caffeine.newBuilder()
                .maximumSize(500)
                .expireAfterWrite(5, TimeUnit.MINUTES)
                .recordStats());
        cacheManager.setAllowNullValues(false);
        return cacheManager;
    }

    @Bean
    @ConfigurationProperties(prefix = "ml.service")
    public MlServiceProperties mlServiceProperties() {
        return new MlServiceProperties();
    }

    public static class MlServiceProperties {
        private String url = "http://ml-service:8001";

        public String getUrl() {
            return url;
        }

        public void setUrl(String url) {
            this.url = url;
        }
    }
}
