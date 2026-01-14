from modules.pipeline.pipeline import Pipeline

CONFIG = {
    'VIDEO_PATH': "./data/clips/easy2.mp4",
    'OUTPUT_VIDEO_PATH': "./data/output/result_easy2.mp4",
    'ELEMENTS_PATH': "./data/elements/circles",
    # 'ELEMENTS_PATH2': "./data/elements/dices",
    
    'MIN_SIDE': 500,
    'MAX_SIDE': 1000,
    'TARGET_WARPED_SIZE': 600,
    
    'HISTORY_LEN': 25,
    'CONFIDENCE_THRESH': 15,
    'PERSISTENCE_THRESH': 15,
    'PERSISTENCE_FRAMES': 20,
    'SKIP_FRAMES': 10,

    'DIFFICULTY' : 'MEDIUM'
}

if __name__ == "__main__":
    pipeline = Pipeline(CONFIG)
    pipeline.run()