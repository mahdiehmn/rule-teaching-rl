"""Teacher-free scheduling for the prospective advisor sham control."""


def validate_sham(args, auxiliary=None):
    """
    Refuse modes whose scheduling or updates require teacher output.
    """

    if not args.advisor_sham:
        return
    if not args.guidance or not args.advisor_rng_isolation:
        raise ValueError('advisor_sham requires guidance and RNG isolation')
    if args.advisor not in ('unlimited', 'early', 'importance'):
        raise ValueError('advisor_sham needs a teacher-free advisor gate')
    if (
        args.peek != 'none' or args.importance_source == 'teacher_q'
        or args.teacher_stream or args.action_reference
        or args.explanation != 'none' or args.consequence != 'none'
        or args.advice_replay or args.imitation_weighting != 'none'
        or args.advice_alias_rate or args.advice_evidence_gate != 'none'
        or args.advice_keep_fraction != 1.0 or args.teacher_evidence
        or args.audit_explanations or args.budget_ledger
        or auxiliary is not None
    ):
        raise ValueError('advisor_sham cannot use teacher-dependent options')


class ShamAdvisor:
    """
    Run the real advisor's gates and pacing with virtual slot counts.

    Only teacher-free gates are supported. An accepted slot advances
    the scheduler as a successful delivery would, but no teacher,
    Advice object, target or cost is created. These virtual counts
    never masquerade as consultations in the trainer or its summary.
    """

    num_asked = 0

    def __init__(self, scheduler):
        """
        Keep scheduler state separate from actual teacher counters.
        """

        if scheduler.name not in ('unlimited', 'early', 'importance'):
            raise ValueError('ShamAdvisor requires a teacher-free gate')
        self.scheduler = scheduler
        self.first_selected_step = None
        self.last_selected_step = None

    def note_env_steps(self, count):
        """
        Preserve the real pacing controller's environment clock.
        """

        self.scheduler.note_env_steps(count)

    def note_step(self, step):
        """
        Preserve the real advisor's visible-step bookkeeping.
        """

        self.scheduler.note_step(step)

    def schedule(self, state, student_action, context, global_step):
        """
        Exercise the pre-consultation path without requesting a label.
        """

        scheduler = self.scheduler
        scheduler.num_steps += 1
        scheduler._pace()
        importance = float(scheduler.importance_fn(state, context))
        scheduler._importance_sum += importance
        scheduler.threshold_ceiling = max(
            scheduler.threshold_ceiling, importance)
        accepted, reason = scheduler._check_ask(
            state, student_action, importance, context)
        if accepted:
            # Virtual delivery maintains both finite budgets and
            # delivery-based pacing, without constructing a label.
            scheduler.num_asked += 1
            scheduler.num_delivered += 1
            scheduler._importance_spent_sum += importance
            if self.first_selected_step is None:
                self.first_selected_step = global_step
            self.last_selected_step = global_step
        scheduler._finish(
            importance=importance, asked=False, delivered=False,
            student_action=student_action, teacher_action=None,
            reason='sham_selected' if accepted else reason)
        return accepted

    def stats(self):
        """
        Report actual zeros and explicitly named virtual counters.
        """

        stats = self.scheduler.stats()
        stats['sham'] = {
            'slots_seen': self.scheduler.num_steps,
            'slots_selected': self.scheduler.num_asked,
            'slots_declined': (
                self.scheduler.num_steps - self.scheduler.num_asked),
            'first_selected_global_step': self.first_selected_step,
            'last_selected_global_step': self.last_selected_step,
            'virtual_query_budget_remaining': (
                max(0, self.scheduler.query_budget
                    - self.scheduler.num_asked)
                if self.scheduler.query_budget else None),
            'virtual_advice_budget_remaining': (
                max(0, self.scheduler.advice_budget
                    - self.scheduler.num_delivered)
                if self.scheduler.advice_budget else None),
        }
        for key in ('num_asked', 'num_delivered', 'query_rate',
                    'delivery_rate', 'query_rate_env', 'delivery_rate_env',
                    'mean_importance_spent'):
            stats[key] = 0
        return stats
