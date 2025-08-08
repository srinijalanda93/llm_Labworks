#The lab describe about the fetch the feature from the video 
'''which is stored in jk1997 bucket video.mp in calendar-chatbot-project
where need to enable api's followed the service-account in cerdientals
'''

import os
from google.cloud import videointelligence_v1 as videointelligence

# Set your Google Cloud service account credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/srinija/Downloads/calendar-chatbot-project-4952ba51de3c.json"

def analyze_video(bucket_name, video_file):
    try:
        # Create a Video Intelligence API client
        video_client = videointelligence.VideoIntelligenceServiceClient()
        bucket_uri = f"gs://{bucket_name}/{video_file}"

        # Select features to extract (here: LABEL_DETECTION)
        features = [videointelligence.Feature.LABEL_DETECTION]

        print("🎥 Starting video annotation...")
        operation = video_client.annotate_video(
            request={"features": features, "input_uri": bucket_uri}
        )

        # Wait for the operation to complete
        result = operation.result(timeout=300)
        print("---- Video processing complete!")

        # Process label annotations
        annotation_results = result.annotation_results[0]
        for label in annotation_results.segment_label_annotations:
            print(f"\n🔍 Label: {label.entity.description}")

            for segment in label.segments:
                start_time = (
                    segment.segment.start_time_offset.seconds +
                    segment.segment.start_time_offset.microseconds / 1e6
                )
                end_time = (
                    segment.segment.end_time_offset.seconds +
                    segment.segment.end_time_offset.microseconds / 1e6
                )
                confidence = segment.confidence
                print(f"⏱ Time: {start_time:.2f}s to {end_time:.2f}s | Confidence: {confidence:.2f}")

    except Exception as e:
        print(" ERROR:", str(e))


# Call the function
analyze_video("jk1997", "video.mp4")
