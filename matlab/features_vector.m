
features_raw=[1 % r_peak
          0 % r_peak_value
          0 %rr_pre_interval
          0 %rr_post_interval
          0 %p_onset
          0 %p_onset_val
          1 %p_peak
          1 %p_peak_va
          0 %p_end
          1 %p_end_va
          1 %qrs_onset
          0 %qrs_onset_val
          0 %qrs_end
          0 %qrs_end_val
          0 %t_peak
          0 %t_peak_val
          0 %t_end
          0 ];%t_end_val
      
      
features.val=features_raw;
features.nam= {'r_peak '
          'r_peak_value '
          'rr_pre_interval '
          'rr_post_interval '
          'p_onset '
          'p_onset_val '
          'p_peak '
          'p_peak_va '
          'p_end '
          'p_end_va '
          'qrs_onset '
          'qrs_onset_val '
          'qrs_end '
          'qrs_end_val '
          't_peak '
          't_peak_val '
          't_end '
          't_end_val '};