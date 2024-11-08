#include "CachingAllocatorConfig.h"

size_t CachingAllocatorConfig::max_split_size() { return instance().m_max_split_size; }

double CachingAllocatorConfig::garbage_collection_threshold() { return instance().m_garbage_collection_threshold; }

bool CachingAllocatorConfig::expandable_segments() { return instance().m_expandable_segments; }

double CachingAllocatorConfig::default_lc_threshold() { return instance().m_default_lc_threshold; }

bool CachingAllocatorConfig::open_memory_optimize() { return instance().m_open_memory_optimize; }

CachingAllocatorConfig::CachingAllocatorConfig()
        :  m_max_split_size(std::numeric_limits<size_t>::max()),
           m_garbage_collection_threshold(0),
           m_expandable_segments(false),
           m_default_lc_threshold(0),
           m_open_memory_optimize(false) {}

CachingAllocatorConfig& CachingAllocatorConfig::instance() {
    static CachingAllocatorConfig *s_instance = ([]() {
        auto inst = new CachingAllocatorConfig();
        const char* env = getenv("PYTORCH_NPU_ALLOC_CONF");
        inst->parseArgs(env);
        return inst;
    })();
    return *s_instance;
}

void CachingAllocatorConfig::lexArgs(const char* env, std::vector<std::string>& config) {
    std::vector<char> buf;

    size_t env_length = strlen(env);
    for (size_t i = 0; i < env_length; i++) {
        if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
            if (!buf.empty()) {
                config.emplace_back(buf.begin(), buf.end());
                buf.clear();
            }
            config.emplace_back(1, env[i]);
        } else if (env[i] != ' ') {
            buf.emplace_back(static_cast<char>(env[i]));
        }
    }
    if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
    }
}

void CachingAllocatorConfig::consumeToken(const std::vector<std::string>& config, size_t i, const char c) {
    TORCH_CHECK(i < config.size() && config[i].compare(std::string(1, c)) == 0,
            "Error parsing CachingAllocator settings, expected ", c);
}

size_t CachingAllocatorConfig::parseMaxSplitSize(const std::vector<std::string>& config, size_t i) {
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        size_t val1 = 0;
        try{
            val1 = static_cast<size_t>(stoi(config[i]));
        } catch (const std::invalid_argument& e){
            TORCH_CHECK(false, "Error, expecting digital string in config");
        } catch (const std::out_of_range& e){
            TORCH_CHECK(false, "Error, out of int range");
        }
        TORCH_CHECK(val1 > kLargeBuffer / kUnitMB, "CachingAllocator option max_split_size_mb too small, must be > ",
                kLargeBuffer / kUnitMB);
        val1 = std::max(val1, kLargeBuffer / kUnitMB);
        val1 = std::min(val1, (std::numeric_limits<size_t>::max() / kUnitMB));
        m_max_split_size = val1 * kUnitMB;
    } else {
        TORCH_CHECK(false, "Error, expecting max_split_size_mb value");
    }
    return i;
}

size_t CachingAllocatorConfig::parseGarbageCollectionThreshold(const std::vector<std::string>& config, size_t i) {
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        double val1 = 0.0;
        try {
            val1 = stod(config[i]);
        } catch (const std::invalid_argument& e){
            TORCH_CHECK(false, "Error, expecting digital string in config");
        } catch (const std::out_of_range& e) {
            TORCH_CHECK(false, "Error, out of double range");
        }
        TORCH_CHECK(val1 > 0, "garbage_collect_threshold too small, set it 0.0~1.0");
        TORCH_CHECK(val1 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0");
        m_garbage_collection_threshold = val1;
    } else {
        TORCH_CHECK(false, "Error, expecting garbage_collection_threshold value");
    }
    return i;
}

size_t CachingAllocatorConfig::parseExpandableSegments(const std::vector<std::string>& config, size_t i) {
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        TORCH_CHECK(i < config.size() && (config[i] == "True" || config[i] == "False"),
                "Expected a single True/False argument for expandable_segments");
        m_expandable_segments = (config[i] == "True");
        void* ptr = nullptr;
        constexpr size_t virtual_mem_size = 512;
        TORCH_CHECK(aclrtReserveMemAddress(&ptr, virtual_mem_size, 0, NULL, 1) == ACL_ERROR_NONE, \
                                    "Error, failed to reserve memory address");
        TORCH_CHECK(aclrtReleaseMemAddress(ptr) == ACL_ERROR_NONE, \
                                    "Error, failed to release memory address");
    } else {
        TORCH_CHECK(false, "Error, expecting expandable_segments value");
    }
    return i;
}

size_t CachingAllocatorConfig::parseDefaultLcThreshold(const std::vector<std::string> &config, size_t i) {
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        double val1 = 0.0;
        try {
            val1 = stod(config[i]);
        } catch (const std::invalid_argument& e){
            TORCH_CHECK(false, "Error, expecting digital string in config");
        } catch (const std::out_of_range& e) {
            TORCH_CHECK(false, "Error, out of double range");
        }
        TORCH_CHECK(val1 >= 0, "default_lc_threshold too small, set it 0.0~INF");
        m_default_lc_threshold = val1;
    } else {
        TORCH_CHECK(false, "Error, expecting default_lc_threshold value");
    }
    return i;
}

size_t CachingAllocatorConfig::parseOpenMemoryOptimize(const std::vector<std::string> &config, size_t i) {
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        if (config[i] == "true" || config[i] == "1") {
            m_open_memory_optimize = true;
        } else if (config[i] == "false" || config[i] == "0") {
            m_open_memory_optimize = false;
        } else {
            TORCH_CHECK(false, "Error, open_memory_optimize should be true or false or 1 or 0");
        }
    } else {
        TORCH_CHECK(false, "Error, expecting open_memory_optimize value");
    }
    return i;
}

void CachingAllocatorConfig::parseArgs(const char* env) {
    // If empty, set the default values
    m_max_split_size = std::numeric_limits<size_t>::max();
    m_garbage_collection_threshold = 0;
    m_default_lc_threshold = 0;
    m_open_memory_optimize = false;

    if (env == nullptr) {
        return;
    }

    std::vector<std::string> config;
    lexArgs(env, config);

    for (size_t i = 0; i < config.size(); i++) {
        if (config[i].compare("max_split_size_mb") == 0) {
            i = parseMaxSplitSize(config, i);
        } else if (config[i].compare("garbage_collection_threshold") == 0) {
            i = parseGarbageCollectionThreshold(config, i);
        } else if (config[i] == "expandable_segments") {
            i = parseExpandableSegments(config, i);
        } else if (config[i].compare("default_lc_threshold") == 0) {
            i = parseDefaultLcThreshold(config, i);
        } else if (config[i].compare("open_memory_optimize") == 0) {
            i = parseOpenMemoryOptimize(config, i);
        } else {
            TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", config[i]);
        }

        if (i + 1 < config.size()) {
            consumeToken(config, ++i, ',');
        }
    }
}
