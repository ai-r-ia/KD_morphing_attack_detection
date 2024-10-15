#!/bin/bash

# Navigate to the scores directory
cd logs/scores/

# Loop through each subdirectory
for dir in */; do
	        
	        # Ensure that we can change into the subdirectory
		    if cd "$dir"; then

	    echo "Entering directory: $dir"
			            # Look for files named "greedy.npy"
				            find . -type f -name "greedy.npy" | while read -r filepath; do
					                # Print what would be renamed
							            echo "Would rename: $filepath to ${filepath%.npy}_feret.npy"
								            done
									            # Go back to the parent directory
										            cd .. || { echo "Failed to return to parent directory"; exit 1; }
											        else
													        echo "Could not enter directory: $dir"
														    fi
													    done
#find . -type f -name "greedy.npy" | while read filepath; do
    # Rename the file by appending _feret before .npy
 #       mv "$filepath" "${filepath%.npy}_feret.npy"
#	    echo "Renamed $filepath to ${filepath%.npy}_feret.npy"
#done
