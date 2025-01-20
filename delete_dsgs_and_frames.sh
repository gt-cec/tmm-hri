# removes all the DSG files and visualization frames in an episode's directory
# Usage: delete_dsgs_and_frames.sh <episode_id>

# Check for correct number of arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: delete_dsgs_and_frames.sh <episode_id>"
    exit 1
fi

# Check if episode exists
if [ ! -d "episodes/$1" ]; then
    echo "Episode $1 does not exist"
    exit 1
fi

# Remove all detection pickles
rm episodes/$1/DSGs/*
echo "Deleted all DSGs in episodes/$1"
rm visualization_frames/*
echo "Deleted all visualization frames in episodes/$1"
exit 0