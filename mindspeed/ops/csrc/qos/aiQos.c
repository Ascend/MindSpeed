#ifdef __cplusplus
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pybind11/pybind11.h>
#include "dcmi_interface_api.h"

#define NPU_OK (0)
#define MAX_CARD_NUM (16)

#ifndef DCMI_QOS_CFG_RESERVED_LEN
#define DCMI_QOS_CFG_RESERVED_LEN 16
#endif

int set_gbl_qos(int card_id, int device_id, int mode)
{
    struct dcmi_qos_gbl_config gblCfg = { 0 };
    gblCfg.enable = 1;
    gblCfg.autoqos_fuse_en = 1;
    gblCfg.mpamqos_fuse_mode = mode;

    int ret = dcmi_set_device_info(
        card_id,
        device_id,
        DCMI_MAIN_CMD_QOS,
        (unsigned int)DCMI_QOS_SUB_GLOBAL_CONFIG,
        (const void*)(&gblCfg),
        (unsigned int)sizeof(struct dcmi_qos_gbl_config)
    );
    if (ret != 0) {
        printf("[ERROR] Failed to set QoS global configuration for card %d - device %d, error code: %d\n",
            card_id, device_id, ret);
        return ret;
    } else {
        printf("[SUCCESS] Succeeded in setting QoS global configuration for card %d - device %d, mode = %d \n",
            card_id, device_id, mode);
    }

    return 0;
}

int set_bw(int target, unsigned int bw_low, unsigned int bw_high, int hardlimit, int card_id, int device_id)
{
    if (card_id < 0 || device_id < 0) {
        printf("invalid card_id(%d) or device_id(%d)\n", card_id, device_id);
        return -1;
    }
    if (hardlimit < 0 || hardlimit > 1) {
        printf("hardlimit must be 0 or 1\n");
        return -1;
    }

    struct dcmi_qos_mata_config mataCfg = { 0 };

    mataCfg.mpamid = target;
    mataCfg.bw_high = bw_high;
    mataCfg.bw_low = bw_low;
    mataCfg.hardlimit = hardlimit;

    for (int i = 0; i < DCMI_QOS_CFG_RESERVED_LEN; i++) {
        mataCfg.reserved[i] = 0;
    }

    int ret = dcmi_set_device_info(
        card_id,
        device_id,
        DCMI_MAIN_CMD_QOS,
        (unsigned int)DCMI_QOS_SUB_MATA_CONFIG,
        (const void*)(&mataCfg),
        (unsigned int)sizeof(struct dcmi_qos_mata_config)
    );
    if (ret != 0) {
        printf("[card:%d, dev:%d] set mata qos config failed, ret = %d\n",
            card_id, device_id, ret);
        return ret;
    }

    printf("[card:%d, dev:%d] set mata qos config success\n", card_id, device_id);
    return 0;
}

int get_device_id_in_card()
{
    int ret;
    int device_id_max = 0;
    int mcu_id = 0;
    int cpu_id = 0;
    int card_id = 0;
    ret = dcmi_get_device_id_in_card(card_id, &device_id_max, &mcu_id, &cpu_id);
    if (ret != 0) {
        printf("[dev:%d]set mata qos config failed, ret = %d\n", 0, ret);
    }

    printf("card:%d 的device：%d\n", card_id, device_id_max);
    return 0;
}

int get_card_list(int *card_num, int *card_list, int max_len)
{
    if (card_num == NULL || card_list == NULL || max_len <= 0) {
        printf("get_card_list: invalid input parameter\n");
        return -1;
    }

    *card_num = 0;
    for (int i = 0; i < max_len; i++) {
        card_list[i] = 0;
    }

    int ret = dcmi_get_card_list(card_num, card_list, max_len);
    if (ret != 0) {
        printf("get card list fail, ret = %d\n", ret);
        return ret;
    }

    return 0;
}

int qos_init()
{
    int ret = dcmi_init();
    if (ret != NPU_OK) {
        printf("Failed to init dcmi.\n");
        return ret;
    }

    return 0;
}

namespace py = pybind11;

PYBIND11_MODULE(aiQos, m)
{
    m.def(
        "set_bw",
        &set_bw,
        py::arg("target"),
        py::arg("bw_low"),
        py::arg("bw_high"),
        py::arg("hardlimit"),
        py::arg("card_id"),
        py::arg("device_id")
    );

    m.def(
        "init",
        &qos_init);

    m.def(
        "set_gbl_qos",
        &set_gbl_qos,
        py::arg("card_id"),
        py::arg("device_id"),
        py::arg("mode")
    );
}