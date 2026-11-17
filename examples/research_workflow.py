"""Infrastructure example: configurable calls and observations, without a cognitive model."""

class ResearchWorkflow:
    def __init__(self, config):
        self.model = config.get('model', 'default')
        self.options = config.get('options', {})
        self.history_limit = config.get('history_limit', 20)

    def run(self, ctx):
        previous = ctx.state.get('observations', default={'turns': 0, 'last_time': None})
        ctx.observe('previous_time', previous['last_time'])
        ctx.state.set('observations', {'turns': previous['turns'] + 1, 'last_time': ctx.now})

        # Researchers can replace this input selection and call sequence directly.
        messages = ctx.assemble(history_limit=self.history_limit)
        reply = ctx.models.chat(messages, model=self.model, options=self.options, purpose='reply')
        return reply
