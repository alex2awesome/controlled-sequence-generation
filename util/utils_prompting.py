class PromptGenerator():
    def __init__(self, prompt_type=None):
        self.headline_tag = '<HEADLINE>'
        self.label_tag = '<LABELS>'
        self.text_tag = '<TEXT>'
        self.go_tag = '<GO>'
        self.label_format = '<LABEL: %s>'
        self.label_map = {'': 'Error'}
        prompt_mapper = {
            'future': self.future_aware,
            'past': self.past_aware,
            'baseline': self.baseline
        }
        self.prompt_method = prompt_mapper[prompt_type] if prompt_type is not None else None

    @property
    def tokens_to_add(self):
        return ['|||', self.headline_tag, self.label_tag, self.text_tag, self.go_tag]

    @property
    def num_added_tokens(self):
        return len(self.tokens_to_add)

    def resize_tokenizer(self, tokenizer):
        to_add = self.tokens_to_add
        tokenizer.add_tokens(to_add)
        return tokenizer

    def clean_label(self, label):
        label = self.label_map.get(label, label)
        return label.replace('_', ' ')

    def generate_prompt(self, headline, sentences, labels, s_idx):
        labels = list(map(self.clean_label, labels))
        prior_labels = ' ||| '.join(labels)
        prior_text = ' '.join(sentences[:s_idx])
        prompt = ' '.join([
            self.headline_tag, headline,
            self.label_tag, prior_labels,
            self.text_tag, prior_text
        ])
        return prompt

    def future_aware(self, labels, s_idx):
        return labels

    def past_aware(self, labels, s_idx):
        return labels[:s_idx + 1]

    def baseline(self, labels, s_idx):
        return labels[s_idx]

    def generate_prompt_full(self, headline, sentences, labels, s_idx):
        labels_to_use = self.prompt_method(labels, s_idx)
        return self.generate_prompt(headline, sentences, labels_to_use, s_idx)