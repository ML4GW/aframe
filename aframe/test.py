import law
import luigi


class SomeCommonDependency(law.LocalWorkflow):
    verbose = luigi.BoolParameter(default=False)

    def create_branch_map(self):
        return {0: "foo", 1: "bar", 2: "baz"}

    def output(self):
        return law.LocalFileTarget(
            "/home/ethan.marx/projects/aframev2/first_{}.txt".format(
                self.branch
            )
        )

    def run(self):
        self.output().dump(self.branch_data, formatter="text")


class Task(law.Task):
    verbose = luigi.BoolParameter(default=False)

    def requires(self):
        return SomeCommonDependency.req(self)

    def output(self):
        return law.LocalFileTarget(
            "/home/ethan.marx/projects/aframev2/final.txt"
        )

    def run(self):
        if self.verbose:
            print("verbose")
        self.output().dump(
            self.input()["collection"][0].load(), formatter="text"
        )
