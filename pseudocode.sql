START
-- Load trained CNN model for fatigue detection
Load CNN fatigue detection model
-- Prompt user to select a video file
Wait for user to select a video file
IF video file is selected THEN
    Open the video for reading frames

    WHILE video has frames AND user has not clicked stop:
        Read the next frame from video

        -- Detect face using MediaPipe
        Detect face in the frame
        IF face is detected THEN
            -- Prepare input: grayscale face, dotmap, landmarks
            Preprocess the face for CNN input
            -- Predict fatigue using CNN model
            Predict fatigue status using the model
            IF predicted as "Fatigued" THEN
                -- Count this frame towards fatigue detection
                Increment fatigue frame counter
                
        -- Every 1 second, check fatigue condition
        IF 1 second has passed THEN
            IF fatigue count ≥ 2 seconds THEN
                Display: "Fatigued"
            ELSE
                Display: "Not Fatigued"

    -- Done processing video
    Release the video and stop playback
END
