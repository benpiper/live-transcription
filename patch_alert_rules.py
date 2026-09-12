import re

with open("alert_rules.py", "r") as f:
    content = f.read()

# Change __post_init__
content = content.replace('''    def __post_init__(self):
        """Validate and compile regex if needed."""
        if not self.enabled:
            return

        if self.match_type == "regex" and isinstance(self.pattern, str):
            try:
                self._compiled_regex = re.compile(self.pattern, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern in rule '{self.id}': {e}")''', '''    def __post_init__(self):
        """Validate and compile regex if needed."""
        if self.match_type == "regex" and isinstance(self.pattern, str):
            try:
                self._compiled_regex = re.compile(self.pattern, re.IGNORECASE)
            except re.error as e:
                if self.enabled:
                    raise ValueError(f"Invalid regex pattern in rule '{self.id}': {e}")
        elif self.match_type == "word":
            keywords = [self.pattern] if isinstance(self.pattern, str) else self.pattern
            if keywords:
                # Sort keywords by length descending to match longer words first
                keywords = sorted(keywords, key=len, reverse=True)
                pattern = r"\\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")\\b"
                self._compiled_regex = re.compile(pattern, re.IGNORECASE)''')

# Change get_compiled_regex
content = content.replace('''    def get_compiled_regex(self) -> Optional[re.Pattern]:
        """Get compiled regex, compiling if needed."""
        if self._compiled_regex is None and self.match_type == "regex" and isinstance(self.pattern, str):
            try:
                self._compiled_regex = re.compile(self.pattern, re.IGNORECASE)
            except re.error:
                return None
        return self._compiled_regex''', '''    def get_compiled_regex(self) -> Optional[re.Pattern]:
        """Get compiled regex, compiling if needed."""
        if self._compiled_regex is None:
            if self.match_type == "regex" and isinstance(self.pattern, str):
                try:
                    self._compiled_regex = re.compile(self.pattern, re.IGNORECASE)
                except re.error:
                    return None
            elif self.match_type == "word":
                keywords = [self.pattern] if isinstance(self.pattern, str) else self.pattern
                if keywords:
                    keywords = sorted(keywords, key=len, reverse=True)
                    pattern = r"\\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")\\b"
                    self._compiled_regex = re.compile(pattern, re.IGNORECASE)
        return self._compiled_regex''')

# Change matches
content = content.replace('''        elif self.match_type == "word":
            # Word boundary matching - prevent "fire" matching "firearm"
            keywords = [self.pattern] if isinstance(self.pattern, str) else self.pattern
            for kw in keywords:
                pattern = r"\\b" + re.escape(kw) + r"\\b"
                if re.search(pattern, text, re.IGNORECASE):
                    matched_text = text
                    break''', '''        elif self.match_type == "word":
            # Word boundary matching - prevent "fire" matching "firearm"
            regex = self.get_compiled_regex()
            if regex and regex.search(text):
                matched_text = text''')

with open("alert_rules.py", "w") as f:
    f.write(content)
