def update_config(config_dict, map_name):
    if config_dict["env"] == "particle":
        if map_name.split("_")[0] == "spread":
            config_dict["env_args"]["num_agents"] = int(map_name.split("_")[1])
            config_dict["env_args"]["num_landmarks"] = int(map_name.split("_")[2])
            config_dict["env_args"]["agent_view_radius"] = 1.5
        elif map_name.split("_")[0] == "tag" or map_name.split("_")[0] == "htag":
            # 注意：adversaries 是模型控制的智能体
            config_dict["env_args"]["num_adversaries"] = int(map_name.split("_")[1])
            config_dict["env_args"]["num_good_agents"] = int(map_name.split("_")[2])
            config_dict["env_args"]["num_landmarks"] = int(map_name.split("_")[3])
            config_dict["env_args"]["agent_view_radius"] = 1.5
        else:
            assert False, "unsupported map_name"

    if 'use_token' not in config_dict:
        config_dict["use_token"] = False

    if config_dict["use_token"]:
        if config_dict["env"] == "sc2":
            config_dict["env_args"]["obs_align"] = True
            config_dict["env_args"]["state_align"] = True
            config_dict["env_args"]["unit_type_align"] = True
            config_dict["env_args"]["shield_align"] = True

    if 'entity_scheme' in config_dict["env_args"]:
        config_dict["entity_scheme"] = config_dict["env_args"]["entity_scheme"]
    else:
        config_dict["entity_scheme"] = False

    return config_dict
