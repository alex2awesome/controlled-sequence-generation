import torch


def summarize_attributions(attributions):
	attributions = attributions.sum(dim=-1).squeeze(0)  # same as LIG
	attributions = attributions / torch.norm(attributions)
	return attributions


def run_dig_explanation(dig_func, all_input_embed, position_embed, attention_mask, steps, type_embed=None, label=None):
	attributions, delta = dig_func.attribute(
		scaled_features=all_input_embed,
		target=label,
		additional_forward_args=(attention_mask, position_embed, type_embed),
		n_steps=steps,
		return_convergence_delta=True
	)
	attributions_word	= summarize_attributions(attributions)
	return attributions_word, delta


def make_visualization(attribution, delta, prediction, pred_label, true_label, target_label, text, label_idx_mapper):
	from captum.attr import visualization

	# storing couple samples in an array for visualization purposes
	return visualization.VisualizationDataRecord(
		attribution,
		prediction,
		label_idx_mapper[pred_label],
		label_idx_mapper[true_label],
		label_idx_mapper[target_label],
		attribution.sum(),
		text,
		delta
	)