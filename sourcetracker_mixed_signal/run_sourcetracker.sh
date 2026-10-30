#!/bin/bash
# Run SourceTracker2 for mixed_signal experiment
# Source = skin samples (individuals that appear in sink pairs)
# Sink = surface_touched samples
sourcetracker2 gibbs \
    -i otu_table.tsv \
    -m mapping.tsv \
    --output_dir st2_output/ \
    --source_sink_column SourceSink \
    --source_column_value source \
    --sink_column_value sink \
    --source_category_column individual

