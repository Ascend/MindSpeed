#ifndef NPU_CACHE_ALLOCATOR_RECORDER_H
#define NPU_CACHE_ALLOCATOR_RECORDER_H
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <climits>
#include <vector>
#include <set>

enum struct LifeCycleType : uint64_t {
    DEFAULT_LC = 0,
    FIRST_STEP_LC,
    LONG_LC,
};

extern bool g_record_flag;  // is the record in the forward or not
extern size_t g_record_cnt;
extern bool is_precise_match;  // record whether it matches precisely
extern unsigned int pool_idx;

constexpr size_t kMinBlockSize = 512; // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576; // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152; // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520; // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152; // round up large allocs to 2 MB
constexpr size_t kSizeLimit = 1395864371;   // 1.3G = 1395864371 1.5G=1610612736
constexpr size_t myMaxSplitSize = 1800000000;   // 1.67G
constexpr size_t kUnitMB = 1024 * 1024;        // 1MiB = 1024 * 1024 bytes

// Check if it is in the forward stage
static bool _check() { return g_record_flag; }

// Record tense information during the forward phase
void set_g_record_flag(bool);
void set_is_precise_match(bool);

struct MemoryRecorder {
    static std::mutex lock; // protect the static value
    static std::unordered_set<MemoryRecorder *> recorder_set;

    long lc_id_cnt;

    // Record the tensor in the forward phase of RecEle:
    // start creation time, release time, tense size, and whether to release it
    struct RecEle {
        size_t tensor_forward_start_tik; // tensor forward start time
        size_t tensor_forward_end_tik; // tensor forward release time
        size_t size;
        RecEle(size_t tensor_forward_start_tik, size_t tensor_forward_end_tik, size_t size)
                : tensor_forward_start_tik(tensor_forward_start_tik),
                  tensor_forward_end_tik(tensor_forward_end_tik),
                  size(size) {}
        RecEle() = default;
    };

    // specify sorting in the set
    static bool _RecorderComparator(const RecEle a, const RecEle b);

    // forward_rec_set: record the information of tensors in the current forward
    std::set<RecEle, bool (*)(const RecEle, const RecEle)> forward_rec_set;
    // last_forward_rec_set: record the information of the tensor in the previous forward
    std::set<RecEle, bool (*)(const RecEle, const RecEle)> last_forward_rec_set;
    size_t forward_tik; // increase after malloc
    unsigned int forward_count; // record forward count

    // Record whether there is a fork in the current forward stage.
    // If there is a fork, insert the current forward_rec_set into the forward_history array.
    bool has_Another_Situation = false;
    // Forward_history stores possible branches that may arise during the training process.
    std::vector<std::set<RecEle, bool (*)(const RecEle, const RecEle)>> forward_history;

    MemoryRecorder();

    ~MemoryRecorder();

    void add(size_t tensor_forward_start_tik, size_t tensor_forward_end_tik, size_t origin_size);

    void change_forward_end_tik(size_t start_tik, size_t end_tik, size_t origin_size, unsigned int forward_distance,
                                bool in_forward);

    LifeCycleType get_lc(size_t origin_size, size_t *tensor_forward_end, size_t *tensor_forward_start);

    static void start_record();

    void _end_record();

    static void end_record();
};

// Record information for the step stage
struct MallocRecorder {
    static bool in_step_flag;
    static std::mutex lock;
    static std::unordered_set<MallocRecorder *> recorder_set; // Recorder_set records MallocRecorder objects

    long step_lc_id_cnt;

    // Record the start creation time, release time, and size of the tensor in MallocRecorderEle
    struct MallocRecorderEle {
        size_t start_tik;   // tensor alloc start time
        size_t end_tik; // tensor alloc release time
        size_t size;    // round_size

        MallocRecorderEle(size_t start_tik, size_t end_tik, size_t size)
                : start_tik(start_tik), end_tik(end_tik), size(size) {}
    };

    static bool _RecorderComparator(const MallocRecorderEle a, const MallocRecorderEle b);

    // rec_set records the information of tensors in the current step
    std::set<MallocRecorderEle, bool (*)(const MallocRecorderEle, const MallocRecorderEle)> rec_set;
    std::set<MallocRecorderEle, bool (*)(const MallocRecorderEle, const MallocRecorderEle)> last_rec_set;
    size_t tik; // increase after malloc
    unsigned int step_count; // record step times

    // Record whether there is a fork in the current step stage.
    // If there is a fork, insert the current rec_set into the step_history array
    bool has_Another_Situation = false;
    // step_history stores possible branches that may occur during the training process
    std::vector<std::set<MallocRecorderEle, bool (*)(const MallocRecorderEle, const MallocRecorderEle)>> step_history;

    static bool _check();

    void add(size_t start_tik, size_t end_tik, size_t round_size);

    void change_end_tik(size_t start_tik, size_t end_tik, size_t round_size, size_t step_distance, bool in_step);

    bool predict_long(size_t round_size, size_t *tensor_step_end, size_t *tensor_step_start);

    // Determine if there will be a size tensor during the tensor_step_start to tensor_step_end stages.
    bool has_tensor_in_step(size_t tensor_step_start, size_t tensor_step_end, size_t seg_size);

    static size_t get_allocation_size(size_t size, LifeCycleType lc = LifeCycleType::DEFAULT_LC);

    MallocRecorder();

    ~MallocRecorder();

    static void start_record();

    static void end_record();
};


#endif
