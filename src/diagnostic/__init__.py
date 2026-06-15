"""Clinical 12-lead ECG diagnostic track (PTB-XL).

A separate pipeline from the MIT-BIH single-beat classifier in the parent
package. Trains a multi-label diagnostic model on PTB-XL for the five
diagnostic superclasses, in either 12-lead (benchmark) or single-lead
(Lead II, matching the image-digitization demo) configurations.
"""
