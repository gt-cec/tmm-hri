# removes all the detection pickles in an episode's directory
# Usage: delete_detect_pickles.sh <episode_id>

# Check for correct number of arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: delete_detect_pickles.sh <episode_id>"
    exit 1
fi

# Check if episode exists
if [ ! -d "episodes/$1" ]; then
    echo "Episode $1 does not exist"
    exit 1
fi

# Remove all detection pickles
rm episodes/$1/0/*_detections.pkl
rm episodes/$1/1/*_detections.pkl
echo "Deleted all detection pickles in episodes/$1"
exit 0